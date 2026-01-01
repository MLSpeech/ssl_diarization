#!/usr/bin/env python3
"""
End-to-end diarization runner:
1) Barlow embedding extraction (frame-level)
2) PLDA scoring (via Kaldi binaries)
3) Agglomerative clustering
4) RTTM output

This script is designed to work with the existing repo conventions:
- A dataset manifest JSON: list of [wav_path, num_samples]
- Segment wav paths encode a "recording id" as the parent folder name
- Segment wav paths encode a "chunk id" as a trailing "-<int>" in the stem (e.g. chunk-001.wav)
- A VAD segments file per recording: <recording>.segments, containing lines where:
    field[0] = wav path (with chunk id), field[2] = start time (sec), field[3] = end time (sec)
- Kaldi PLDA model directory containing: mean.vec, plda_model
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from barlow_model import Barlow_diarization, window
import libs.aggHC as AHC


@dataclass(frozen=True)
class Segment:
    chunk_id: int
    start_s: float
    end_s: float

def exclude_bias_and_norm(p):
    return p.ndim == 1

def _require_bin(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise RuntimeError(
            f"Required binary '{bin_name}' not found in PATH. "
            f"Install Kaldi binaries or add them to PATH."
        )
    return p


def _parse_chunk_id_from_path(wav_path: str) -> int:
    stem = Path(wav_path).stem
    # Common patterns:
    # - chunk-001
    # - anything-123
    m = re.search(r"(?:^|-)0*(\d+)$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"chunk-0*(\d+)", stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse chunk id from wav path stem='{stem}' (path={wav_path})")


def _default_rec_id_from_path(wav_path: str) -> str:
    # For a single recording, default to the file stem.
    return Path(wav_path).stem


def _load_segments_file(segments_path: Path) -> List[Segment]:
    """
    Legacy mode: parse an existing <rec>.segments file.
    Expected fields: path start end at parts[2], parts[3].
    """
    segs: List[Segment] = []
    for line in segments_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad segments line in {segments_path}: {line}")
        wav = parts[0]
        start_s = float(parts[2])
        end_s = float(parts[3])
        segs.append(Segment(chunk_id=_parse_chunk_id_from_path(wav), start_s=start_s, end_s=end_s))
    segs.sort(key=lambda s: s.chunk_id)
    return segs


def _load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise ValueError(f"Unexpected audio tensor shape {wav.shape} for {path}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)


def _resample_if_needed(wav_1xt: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav_1xt
    resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out)
    return resampler(wav_1xt)

def _silero_vad_segments_from_file(
    wav_path: Path,
    sampling_rate: int = 16000,
    threshold: float = 0.04,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 400,
    speech_pad_ms: int = 80,
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """
    Run Silero VAD and return:
    - audio_1d: 1D tensor at `sampling_rate` (as loaded by Silero's read_audio)
    - segments: list of (start_s, end_s) in seconds
    """
    # Silero VAD is loaded via torch.hub; may download on first run.
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        skip_validation=True,
        trust_repo=True,
    )
    get_speech_timestamps, _, read_audio, _, _ = utils

    audio_1d = read_audio(str(wav_path), sampling_rate=sampling_rate)
    timestamps = get_speech_timestamps(
        audio_1d,
        vad_model,
        sampling_rate=sampling_rate,
        threshold=float(threshold),
        min_speech_duration_ms=int(min_speech_duration_ms),
        min_silence_duration_ms=int(min_silence_duration_ms),
        speech_pad_ms=int(speech_pad_ms),
        return_seconds=True,
    )
    segs = [(float(t["start"]), float(t["end"])) for t in timestamps]
    return audio_1d, segs


def _segments_to_chunk_list(speech_segs_s: List[Tuple[float, float]]) -> List[Segment]:
    return [Segment(chunk_id=i, start_s=s, end_s=e) for i, (s, e) in enumerate(speech_segs_s)]


def _write_vad_chunks_and_segments_file(
    audio_1d: torch.Tensor,
    sampling_rate: int,
    speech_segs_s: List[Tuple[float, float]],
    out_base_wav: Path,
    recording_id: str,
) -> Path:
    """
    Debug/inspection helper: writes chunk WAVs and a Kaldi-ish .segments file with:
      <chunk_path> <recording_id> <start_s> <end_s>
    Returns path to the written .segments file.
    """
    out_dir = Path(str(out_base_wav).replace(".wav", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_path = Path(str(out_base_wav).replace(".wav", ".segments"))

    with segments_path.open("w") as f:
        for i, (start_s, end_s) in enumerate(speech_segs_s):
            s = max(0, int(start_s * sampling_rate))
            e = max(s + 1, int(end_s * sampling_rate))
            chunk_path = out_dir / f"chunk-{i:03d}.wav"
            chunk_1xt = audio_1d[s:e].unsqueeze(0).cpu()
            torchaudio.save(str(chunk_path), chunk_1xt, sampling_rate)
            f.write(f"{chunk_path} {recording_id} {start_s:.6f} {end_s:.6f}\n")
    return segments_path


def _kaldi_write_vec_text(key: str, vec: np.ndarray) -> str:
    # Kaldi text vector format: "utt  [ 1 2 3 ]"
    vals = " ".join(f"{x:.8f}" for x in vec.tolist())
    return f"{key}  [ {vals} ]\n"


def _kaldi_run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _parse_kaldi_text_matrices(path: Path) -> Dict[str, np.ndarray]:
    """
    Parse a Kaldi text ark containing one or more matrices, keyed by utterance.
    Example:
      rec1  [
        0.1 0.2
        0.3 0.4 ]
    """
    out: Dict[str, np.ndarray] = {}
    key: Optional[str] = None
    rows: List[List[float]] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        if key is None:
            # Expect "key [" possibly with content after '['.
            parts = line.split()
            key = parts[0]
            if "[" not in line:
                raise ValueError(f"Unexpected Kaldi ark line (missing '['): {raw}")
            after = line.split("[", 1)[1].strip()
            if after:
                has_end = "]" in after
                after = after.replace("]", "").strip()
                if after:
                    rows.append([float(x) for x in after.split()])
                if has_end:
                    out[key] = np.asarray(rows, dtype=np.float32)
                    key = None
                    rows = []
            continue

        # In-matrix lines
        if "]" in line:
            line = line.replace("]", "").strip()
            if line:
                rows.append([float(x) for x in line.split()])
            out[key] = np.asarray(rows, dtype=np.float32)
            key = None
            rows = []
        else:
            rows.append([float(x) for x in line.split()])

    if key is not None:
        raise ValueError(f"Unterminated matrix for key={key} in {path}")
    return out


def _scores_to_dist(scores: np.ndarray, shift_min_to_zero: bool = True) -> np.ndarray:
    # High PLDA score = more similar; clustering wants distances => negate.
    dist = (-scores).astype(np.float32)
    # Symmetrize and clean diagonal.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    if shift_min_to_zero:
        mn = float(dist.min())
        if mn < 0.0:
            dist = dist - mn
            np.fill_diagonal(dist, 0.0)
    return dist


def _extract_frame_embeddings_for_chunk(
    model: Barlow_diarization,
    audio_1xt: torch.Tensor,
    sample_rate: int,
    frame_step_s: float,
    device: torch.device,
) -> np.ndarray:
    """
    Convert a single chunk waveform into frame embeddings (one per frame_step_s).
    The model architecture stride corresponds to 0.5s; to get 0.25s hop we
    explicitly window into overlapping 0.5s windows and concatenate them.
    """
    # audio_1xt is [1, T]
    y = audio_1xt.squeeze(0).cpu().numpy().astype(np.float32)
    receptive_field = int(sample_rate * 0.5)  # 0.5s
    hop = int(sample_rate * frame_step_s)
    if y.shape[0] <= receptive_field:
        pad_width = receptive_field - y.shape[0] + 1
        y = np.pad(y, (0, pad_width), mode="mean")

    y_win = window(y, receptive_field, hop)
    # barlow_model.window returns a torch.Tensor; barlow_model8k.window returned numpy.
    if isinstance(y_win, torch.Tensor):
        y_win = y_win.detach().cpu().numpy()
    y_win = np.asarray(y_win, dtype=np.float32).reshape(1, -1)  # [1, T_concat]

    # Mirror the legacy scripts: make sure there is enough context.
    min_len = 2 * sample_rate
    if y_win.shape[1] < min_len:
        prepend = y_win[:, : min(sample_rate, y_win.shape[1])]
        y_win = np.concatenate([prepend, y_win], axis=1)

    x = torch.from_numpy(y_win).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _z, z1 = model.embed(x)
    # z1 is [n_frames, emb_dim]
    emb = np.nan_to_num(z1.detach().cpu().numpy()).astype(np.float32)
    return emb


def _load_barlow_model(
    checkpoint_path: Path,
    channels: int,
    fc_dim: int,
    device: torch.device,
    weights_only: bool = False,
) -> Barlow_diarization:
    model = Barlow_diarization(int(channels), int(fc_dim), 1, 0).to(device)
    # torch.load signature changed over time; weights_only exists in newer torch.
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=weights_only)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _read_reco2numspk(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec, n = line.split()[:2]
        out[rec] = int(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Barlow embedding extraction + Kaldi PLDA scoring + clustering to RTTM."
    )
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--wav", type=Path, help="Single input wav to diarize (will run Silero VAD).")
    # Alias for convenience / backwards compatibility with other scripts.
    inp.add_argument("--audio", dest="wav", type=Path, help="Alias for --wav.")
    inp.add_argument("--manifest-json", type=Path, help="Legacy mode: JSON list of [wav_path, num_samples].")

    ap.add_argument("--checkpoint", required=True, type=Path, help="Barlow checkpoint (.pth) containing 'model'.")
    ap.add_argument("--plda-model-dir", required=True, type=Path, help="Dir containing mean.vec and plda_model.")
    ap.add_argument("--out-rttm-dir", required=True, type=Path, help="Output directory for per-recording RTTM files.")

    ap.add_argument(
        "--vad-segments-dir",
        type=Path,
        default=None,
        help="Legacy mode only: directory containing <rec>.segments files.",
    )

    ap.add_argument(
        "--channels",
        type=int,
        default=1024,
        help="Barlow encoder CNN channels (the 'dim' argument used when training).",
    )
    ap.add_argument("--fc-dim", type=int, default=512, help="Barlow projector dim used in the checkpoint.")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Waveform sample-rate expected by the model.")
    ap.add_argument("--frame-step", type=float, default=0.25, help="Frame hop in seconds (RTTM resolution).")
    ap.add_argument("--start-offset", type=float, default=0.0, help="Seconds to add to all start times.")
    ap.add_argument("--num-workers", type=int, default=2, help="(legacy) DataLoader workers for audio I/O.")
    ap.add_argument("--num-speakers", type=int, default=2, help="Number of speakers (default: 2).")
    ap.add_argument("--reco2num-spk", type=Path, default=None, help="Optional per-recording mapping file.")

    ap.add_argument("--silero-threshold", type=float, default=0.04)
    ap.add_argument("--silero-min-speech-ms", type=int, default=200)
    ap.add_argument("--silero-min-silence-ms", type=int, default=400)
    ap.add_argument("--silero-speech-pad-ms", type=int, default=80)
    ap.add_argument(
        "--save-vad-chunks",
        action="store_true",
        help="If set, writes VAD chunk wavs + a .segments file under the work dir (debug/inspection).",
    )

    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate Kaldi files (default: <out-rttm-dir>/.work).",
    )
    ap.add_argument(
        "--shift-dist-min-to-zero",
        action="store_true",
        help="Shift PLDA-derived distances so the minimum value becomes 0 (helps if negatives exist).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")

    args = ap.parse_args()

    args.out_rttm_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or (args.out_rttm_dir / ".work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Validate Kaldi artifacts
    mean_vec = args.plda_model_dir / "mean.vec"
    plda_model = args.plda_model_dir / "plda_model"
    if not mean_vec.exists():
        raise FileNotFoundError(f"Missing {mean_vec}")
    if not plda_model.exists():
        raise FileNotFoundError(f"Missing {plda_model}")

    # Validate required Kaldi binaries
    _require_bin("ivector-subtract-global-mean")
    _require_bin("ivector-plda-scoring-dense")
    _require_bin("copy-feats")

    device = torch.device(args.device)
    model = _load_barlow_model(args.checkpoint, channels=args.channels, fc_dim=args.fc_dim, device=device)

    # Speaker-count config
    reco2num = _read_reco2numspk(args.reco2num_spk) if args.reco2num_spk else {}

    # rec -> chunk_id -> embeddings [n_frames, d]
    chunk_embs: Dict[str, Dict[int, np.ndarray]] = {}
    rec_segments: Dict[str, List[Segment]] = {}

    if args.wav is not None:
        wav_path = args.wav
        if not wav_path.exists():
            raise FileNotFoundError(str(wav_path))

        rec_id = _default_rec_id_from_path(str(wav_path))

        # Use Silero's read_audio + get_speech_timestamps to match typical silero behavior.
        audio_1d_16k, speech_segs_s = _silero_vad_segments_from_file(
            wav_path,
            sampling_rate=16000,
            threshold=args.silero_threshold,
            min_speech_duration_ms=args.silero_min_speech_ms,
            min_silence_duration_ms=args.silero_min_silence_ms,
            speech_pad_ms=args.silero_speech_pad_ms,
        )
        if not speech_segs_s:
            raise RuntimeError(f"Silero VAD returned 0 speech segments for {wav_path}")

        segs = _segments_to_chunk_list(speech_segs_s)
        rec_segments[rec_id] = segs

        # Convert silero audio (1D at 16k) -> mono [1,T] -> resample to model SR for embedding extraction (default 16k).
        wav_1xt_16k = audio_1d_16k.unsqueeze(0)
        wav_model_sr = _resample_if_needed(wav_1xt_16k, 16000, args.sample_rate)

        if args.save_vad_chunks:
            vad_dir = work_dir / "vad"
            vad_dir.mkdir(parents=True, exist_ok=True)
            _write_vad_chunks_and_segments_file(
                audio_1d=audio_1d_16k,
                sampling_rate=16000,
                speech_segs_s=speech_segs_s,
                out_base_wav=vad_dir / f"{rec_id}.wav",
                recording_id=rec_id,
            )

        for seg in segs:
            s = max(0, int(seg.start_s * args.sample_rate))
            e = max(s + 1, int(seg.end_s * args.sample_rate))
            chunk = wav_model_sr[:, s:e]
            emb = _extract_frame_embeddings_for_chunk(
                model=model,
                audio_1xt=chunk,
                sample_rate=args.sample_rate,
                frame_step_s=args.frame_step,
                device=device,
            )
            chunk_embs.setdefault(rec_id, {})[seg.chunk_id] = emb
    else:
        # Legacy manifest+segments mode kept for compatibility with existing scripts.
        if args.vad_segments_dir is None:
            raise ValueError("--vad-segments-dir is required when using --manifest-json")
        import json

        files = json.load(open(args.manifest_json, "r"))
        if not files:
            raise RuntimeError("Empty manifest json.")

        for wav_path, _n in files:
            rec_id = Path(wav_path).parent.name
            chunk_id = _parse_chunk_id_from_path(wav_path)
            wav, sr = _load_audio_mono(Path(wav_path))
            wav_8k = _resample_if_needed(wav, sr, args.sample_rate)
            emb = _extract_frame_embeddings_for_chunk(
                model=model,
                audio_1xt=wav_8k,
                sample_rate=args.sample_rate,
                frame_step_s=args.frame_step,
                device=device,
            )
            chunk_embs.setdefault(rec_id, {})[chunk_id] = emb

        for rec_id in sorted(chunk_embs.keys()):
            seg_path = args.vad_segments_dir / f"{rec_id}.segments"
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing VAD segments file for rec='{rec_id}': {seg_path}")
            rec_segments[rec_id] = _load_segments_file(seg_path)

    # Build Kaldi scoring inputs: ivecs.ark + spk2utt
    score_dir = work_dir / "plda_score"
    score_dir.mkdir(parents=True, exist_ok=True)
    ivecs_path = score_dir / "ivecs.ark"
    spk2utt_path = score_dir / "spk2utt"

    # We also need per-rec timing info for RTTM: segments + frame_counts
    rec_frame_counts: Dict[str, List[int]] = {}
    rec_total_frames: Dict[str, int] = {}

    with ivecs_path.open("w") as ivecs_f, spk2utt_path.open("w") as spk2utt_f:
        for rec_id in sorted(chunk_embs.keys()):
            segs = rec_segments.get(rec_id)
            if not segs:
                raise RuntimeError(f"Missing segments for rec='{rec_id}' (this should not happen).")
            # Concatenate per-segment (chunk) frames in VAD order
            frame_counts: List[int] = []
            key_list: List[str] = []
            total = 0

            for seg in segs:
                if seg.chunk_id not in chunk_embs[rec_id]:
                    raise KeyError(
                        f"rec='{rec_id}' missing chunk_id={seg.chunk_id} in manifest. "
                        f"Either fix the manifest or the .segments file."
                    )
                frames = chunk_embs[rec_id][seg.chunk_id]
                frame_counts.append(int(frames.shape[0]))
                for i in range(frames.shape[0]):
                    utt_key = f"{rec_id}_{total}"
                    ivecs_f.write(_kaldi_write_vec_text(utt_key, frames[i]))
                    key_list.append(utt_key)
                    total += 1

            spk2utt_f.write(rec_id + " " + " ".join(key_list) + "\n")
            rec_frame_counts[rec_id] = frame_counts
            rec_total_frames[rec_id] = total

    # Run Kaldi PLDA scoring
    ivecs_mean = score_dir / "ivecs_mean.ark"
    scores_ark = score_dir / "scores.ark"
    scores_txt = score_dir / "plda_scores_t.ark"

    _kaldi_run(
        [
            "ivector-subtract-global-mean",
            str(mean_vec),
            f"ark:{ivecs_path}",
            f"ark:{ivecs_mean}",
        ]
    )
    _kaldi_run(
        [
            "ivector-plda-scoring-dense",
            str(plda_model),
            f"ark:{spk2utt_path}",
            f"ark:{ivecs_mean}",
            f"ark:{scores_ark}",
        ]
    )
    _kaldi_run(["copy-feats", f"ark:{scores_ark}", f"ark,t:{scores_txt}"])

    score_mats = _parse_kaldi_text_matrices(scores_txt)

    # Cluster per recording and write RTTM
    for rec_id in sorted(rec_segments.keys()):
        if rec_id not in score_mats:
            raise KeyError(
                f"Missing PLDA score matrix for rec='{rec_id}' in {scores_txt}. "
                f"Found keys: {sorted(score_mats.keys())[:10]}..."
            )

        n_frames = rec_total_frames[rec_id]
        mat = score_mats[rec_id]
        if mat.shape[0] != n_frames or mat.shape[1] != n_frames:
            raise ValueError(
                f"Score matrix shape mismatch for rec='{rec_id}': "
                f"expected ({n_frames},{n_frames}) got {mat.shape}"
            )

        dist = _scores_to_dist(mat, shift_min_to_zero=bool(args.shift_dist_min_to_zero))

        n_spk = reco2num.get(rec_id, int(args.num_speakers))

        # Convert Segment dataclass list to the structure expected by aggHC
        segs = [[s.chunk_id, s.start_s, s.end_s] for s in rec_segments[rec_id]]
        rttm_out = args.out_rttm_dir / f"{rec_id}.rttm"

        # NOTE: aggHC expects `frame_counts` for the VAD segments order.
        AHC.n_clusters_plda_vad_segments(
            frame_counts=rec_frame_counts[rec_id],
            embeddings=dist,
            segments=segs,
            N=int(n_spk),
            rttmOut=str(rttm_out),
            filename=rec_id,
            stepSize=float(args.frame_step),
            offset=float(args.start_offset),
            print_labels=False,
            memory=None,
            vad=None,
        )

    print(f"Done. RTTMs written to: {args.out_rttm_dir}")


if __name__ == "__main__":
    main()



