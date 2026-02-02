#!/usr/bin/env python3
"""
Simple end-to-end diarization (single wav):
1) Silero VAD -> speech segments
2) Barlow embeddings (from `barlow_model.py`, usually 16k)
3) Kaldi PLDA dense scoring (mean.vec + plda_model)
4) Agglomerative clustering (precomputed distance)
5) RTTM output
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from barlow_model import Barlow_diarization, window

def exclude_bias_and_norm(p):
    return p.ndim == 1

@dataclass(frozen=True)
class SpeechSeg:
    start_s: float
    end_s: float


def _require_bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Missing required binary on PATH: {name}")
    return p


def _resample(wav_1xt: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav_1xt
    return torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out)(wav_1xt)


def _run_silero_vad(
    wav_path: Path,
    sampling_rate: int = 16000,
    threshold: float = 0.04,
    min_speech_ms: int = 200,
    min_silence_ms: int = 400,
    speech_pad_ms: int = 80,
) -> Tuple[torch.Tensor, List[SpeechSeg]]:
    """
    Returns:
    - audio_1d: 1D float tensor at sampling_rate
    - speech segments in seconds
    """
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        skip_validation=True,
        trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = utils
    audio_1d = read_audio(str(wav_path), sampling_rate=sampling_rate)
    ts = get_speech_timestamps(
        audio_1d,
        vad_model,
        sampling_rate=sampling_rate,
        threshold=float(threshold),
        min_speech_duration_ms=int(min_speech_ms),
        min_silence_duration_ms=int(min_silence_ms),
        speech_pad_ms=int(speech_pad_ms),
        return_seconds=True,
    )
    segs = [SpeechSeg(float(t["start"]), float(t["end"])) for t in ts]
    return audio_1d, segs


def _load_barlow(checkpoint: Path, channels: int, fc_dim: int, device: torch.device) -> Barlow_diarization:
    model = Barlow_diarization(int(channels), int(fc_dim), 1, 0).to(device)
    # permissive across torch versions
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _embed_segment(
    model: Barlow_diarization,
    seg_wav_1xt: torch.Tensor,
    sample_rate: int,
    frame_step_s: float,
    device: torch.device,
) -> np.ndarray:
    """
    Segment waveform -> frame embeddings. Uses repo's `window()` helper.
    """
    y = seg_wav_1xt.squeeze(0).detach().cpu().numpy().astype(np.float32)
    receptive = int(sample_rate * 0.5)
    hop = int(sample_rate * frame_step_s)
    if y.shape[0] <= receptive:
        y = np.pad(y, (0, receptive - y.shape[0] + 1), mode="mean")

    w = window(y, receptive, hop)
    # barlow_model.window returns torch.Tensor
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float32).reshape(1, -1)

    # small-context safety (mirrors old scripts)
    if w.shape[1] < 2 * sample_rate:
        w = np.concatenate([w[:, : min(sample_rate, w.shape[1])], w], axis=1)

    x = torch.from_numpy(w).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, z1 = model.embed(x)
    return np.nan_to_num(z1.detach().cpu().numpy()).astype(np.float32)


def _write_kaldi_ivecs_text(path: Path, utt_ids: List[str], X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u, v in zip(utt_ids, X):
            vals = " ".join(f"{float(x):.8f}" for x in v.tolist())
            f.write(f"{u}  [ {vals} ]\n")


def _write_kaldi_spk2utt(path: Path, reco_key: str, utt_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(reco_key + " " + " ".join(utt_ids) + "\n")


def _parse_kaldi_text_matrices(path: Path) -> Dict[str, np.ndarray]:
    """
    Parse `copy-feats ark,t:` output. Returns dict[key]->matrix.
    """
    out: Dict[str, np.ndarray] = {}
    key: Optional[str] = None
    rows: List[List[float]] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        if key is None:
            parts = line.split()
            key = parts[0]
            if "[" not in line:
                raise ValueError(f"Unexpected kaldi line (missing '['): {raw}")
            after = line.split("[", 1)[1].strip()
            if after:
                has_end = "]" in after
                after = after.replace("]", "").strip()
                if after:
                    rows.append([float(x) for x in after.split()])
                if has_end:
                    out[key] = np.asarray(rows, dtype=np.float32)
                    key, rows = None, []
            continue

        if "]" in line:
            line = line.replace("]", "").strip()
            if line:
                rows.append([float(x) for x in line.split()])
            out[key] = np.asarray(rows, dtype=np.float32)
            key, rows = None, []
        else:
            rows.append([float(x) for x in line.split()])

    if key is not None:
        raise ValueError(f"Unterminated matrix for key={key}")
    return out


def _plda_dense_kaldi(
    X: np.ndarray,
    mean_vec: Path,
    plda_model: Path,
    work_dir: Path,
    reco_key: str = "rec",
) -> np.ndarray:
    """
    Returns dense similarity matrix S (NxN).

    Key point: we always write spk2utt key as 'rec' and parse matrix for 'rec'
    to avoid any key-mismatch issues.
    """
    _require_bin("ivector-subtract-global-mean")
    _require_bin("ivector-plda-scoring-dense")
    _require_bin("copy-feats")

    work_dir.mkdir(parents=True, exist_ok=True)
    ivecs = work_dir / "ivecs.ark"
    spk2utt = work_dir / "spk2utt"
    ivecs_mean = work_dir / "ivecs_mean.ark"
    scores_ark = work_dir / "scores.ark"
    scores_txt = work_dir / "scores.txt"

    utt_ids = [f"utt{i}" for i in range(X.shape[0])]
    _write_kaldi_ivecs_text(ivecs, utt_ids, X)
    _write_kaldi_spk2utt(spk2utt, reco_key, utt_ids)

    subprocess.run(
        ["ivector-subtract-global-mean", str(mean_vec), f"ark:{ivecs}", f"ark:{ivecs_mean}"],
        check=True,
    )
    subprocess.run(
        ["ivector-plda-scoring-dense", str(plda_model), f"ark:{spk2utt}", f"ark:{ivecs_mean}", f"ark:{scores_ark}"],
        check=True,
    )
    subprocess.run(["copy-feats", f"ark:{scores_ark}", f"ark,t:{scores_txt}"], check=True)

    mats = _parse_kaldi_text_matrices(scores_txt)
    if reco_key not in mats:
        raise RuntimeError(f"Kaldi scores missing key '{reco_key}'. Found keys: {sorted(mats.keys())}")
    S = mats[reco_key]
    if S.shape != (X.shape[0], X.shape[0]):
        raise RuntimeError(f"Unexpected score shape {S.shape}, expected ({X.shape[0]},{X.shape[0]})")
    return S


def _scores_to_distance(S: np.ndarray) -> np.ndarray:
    # match legacy: multiply by -1 to make it a distance-like matrix
    D = (-S).astype(np.float32)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    mn = float(D.min())
    if mn < 0.0:
        D = D - mn
        np.fill_diagonal(D, 0.0)
    return D


def _cluster_precomputed(D: np.ndarray, n_spk: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    try:
        model = AgglomerativeClustering(n_clusters=int(n_spk), metric="cosine", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=int(n_spk), affinity="cosine", linkage="average")
    return model.fit_predict(D).astype(int)


def _labels_to_rttm(
    segs: List[SpeechSeg],
    frame_counts: List[int],
    labels: np.ndarray,
    out_rttm: Path,
    reco_id: str,
    frame_step: float,
    start_offset: float = 0.0,
) -> None:
    """
    Writes RTTM where each frame is `frame_step` and consecutive frames with same label are merged.
    Segments are laid out in the VAD segment order.
    """
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    with out_rttm.open("w", encoding="utf-8") as f:
        idx = 0
        for seg, n_frames in zip(segs, frame_counts):
            t = seg.start_s + start_offset
            j = 0
            while j < n_frames:
                lab = int(labels[idx])
                dur = frame_step
                while j < n_frames - 1 and int(labels[idx]) == int(labels[idx + 1]):
                    dur += frame_step
                    j += 1
                    idx += 1
                f.write(f"SPEAKER {reco_id} 0 {t:.3f} {dur:.3f} <NA> <NA> spk{lab} <NA> <NA>\n")
                t += dur
                j += 1
                idx += 1


def main() -> None:
    ap = argparse.ArgumentParser("Simple Barlow+PLDA diarization (single wav)")
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--plda-model-dir", required=True, type=Path, help="Must contain mean.vec + plda_model")
    ap.add_argument("--out-rttm", required=True, type=Path)

    ap.add_argument("--num-speakers", type=int, default=2)
    ap.add_argument("--channels", type=int, default=1024)
    ap.add_argument("--fc-dim", type=int, default=512)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--frame-step", type=float, default=0.25)
    ap.add_argument("--start-offset", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work-dir", type=Path, default=None)

    # silero params
    ap.add_argument("--silero-threshold", type=float, default=0.04)
    ap.add_argument("--silero-min-speech-ms", type=int, default=200)
    ap.add_argument("--silero-min-silence-ms", type=int, default=400)
    ap.add_argument("--silero-speech-pad-ms", type=int, default=80)

    args = ap.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(str(args.audio))
    if not args.checkpoint.exists():
        raise FileNotFoundError(str(args.checkpoint))

    mean_vec = args.plda_model_dir / "mean.vec"
    plda_model = args.plda_model_dir / "plda_model"
    if not mean_vec.exists():
        raise FileNotFoundError(str(mean_vec))
    if not plda_model.exists():
        raise FileNotFoundError(str(plda_model))

    reco_id = args.audio.stem
    work_dir = args.work_dir or (args.out_rttm.parent / ".work")
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = _load_barlow(args.checkpoint, channels=args.channels, fc_dim=args.fc_dim, device=device)

    audio_1d_16k, vad_segs = _run_silero_vad(
        args.audio,
        sampling_rate=16000,
        threshold=args.silero_threshold,
        min_speech_ms=args.silero_min_speech_ms,
        min_silence_ms=args.silero_min_silence_ms,
        speech_pad_ms=args.silero_speech_pad_ms,
    )
    if not vad_segs:
        raise RuntimeError("Silero VAD returned 0 segments.")

    wav_1xt_16k = audio_1d_16k.unsqueeze(0)
    wav_model = _resample(wav_1xt_16k, 16000, args.sample_rate)

    # embeddings per VAD segment
    all_frames: List[np.ndarray] = []
    frame_counts: List[int] = []
    for seg in vad_segs:
        s = int(max(0.0, seg.start_s) * args.sample_rate)
        e = int(max(seg.start_s + 1e-3, seg.end_s) * args.sample_rate)
        seg_wav = wav_model[:, s:e]
        Z = _embed_segment(model, seg_wav, sample_rate=args.sample_rate, frame_step_s=args.frame_step, device=device)
        frame_counts.append(int(Z.shape[0]))
        all_frames.append(Z)

    X = np.concatenate(all_frames, axis=0)
    if X.shape[0] < 2:
        raise RuntimeError("Not enough frames after VAD to score/cluster.")

    # PLDA scoring
    S = _plda_dense_kaldi(
        X,
        mean_vec=mean_vec,
        plda_model=plda_model,
        work_dir=work_dir / "kaldi_plda",
        reco_key="rec",
    )
    D = _scores_to_distance(S)

    # clustering
    labels = _cluster_precomputed(D, n_spk=args.num_speakers)

    # RTTM
    _labels_to_rttm(
        vad_segs,
        frame_counts=frame_counts,
        labels=labels,
        out_rttm=args.out_rttm,
        reco_id=reco_id,
        frame_step=args.frame_step,
        start_offset=args.start_offset,
    )

    print(f"RTTM written: {args.out_rttm}")


if __name__ == "__main__":
    main()


