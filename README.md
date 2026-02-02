## `ssl_diarization`

Self-supervised speaker diarization code accompanying the Interspeech 2022 work: [**Self-supervised Speaker Diarization**](https://arxiv.org/abs/2204.04166).

Pretrained models: [**`MLSpeech/SSL_diar`**](https://huggingface.co/MLSpeech/SSL_diar)

This repo contains research/training scripts plus a **simple end-to-end inference pipeline** that does:

- **Silero VAD** → speech segments  
- **Barlow embedding extraction** (`barlow_model.py`, typically 16 kHz)  
- **PLDA dense scoring** (Kaldi binaries)  
- **Agglomerative clustering** (precomputed distances)  
- **RTTM output**

## Quickstart (end-to-end diarization → RTTM)

### Prerequisites

- **Conda env** with PyTorch + torchaudio + sklearn + numpy.
- **Kaldi binaries on PATH** (`ivector-subtract-global-mean`, `ivector-plda-scoring-dense`, `copy-feats`).
  - This repo provides `path.sh` to set that up (see below).
- **Network access on first run** (Silero VAD is loaded via `torch.hub` and cached).

### 1) Activate env + Kaldi path

Run from this directory:

conda activate ssl_diar
. ./path.sh

`path.sh` currently points to:

```bash
export KALDI_ROOT=/*/*/kaldi
```

If your Kaldi lives elsewhere, edit `path.sh` accordingly.

### 2) Run diarization on one wav

The end-to-end script is `run_barlow_plda_cluster.py` and expects:

- `--audio`: input wav
- `--checkpoint`: Barlow checkpoint (PyTorch `.pth`)
- `--plda-model-dir`: directory with `mean.vec` and `plda_model`
- `--out-rttm`: output RTTM path

Example (using the checkpoint + PLDA model included in this folder):

```bash
python3 run_barlow_plda_cluster.py \
  --audio Callhome/nist_recognition_evaluation_wav16/niel_test/sid00sg1/data/iaaf.wav \
  --checkpoint ./checkpoint_24.pth \
  --plda-model-dir ./iter_67_plda \
  --out-rttm ./out_rttm_iaaf/iaaf_e2e.rttm \
  --channels 1024 \
  --fc-dim 512 \
  --num-speakers 2
```

Output:

- **RTTM** written to the `--out-rttm` path you provide.
- Intermediate Kaldi artifacts go under `--out-rttm`’s parent directory in `./.work/` unless you override `--work-dir`.

## Script reference: `run_barlow_plda_cluster.py`

Key options:

- **`--num-speakers`**: number of speakers (default `2`).
- **`--channels` / `--fc-dim`**: must match your checkpoint architecture.
- **`--sample-rate`**: model sample rate (default `16000` for `barlow_model.py`).
- **Silero VAD knobs**:
  - `--silero-threshold` (default `0.04`)
  - `--silero-min-speech-ms` (default `200`)
  - `--silero-min-silence-ms` (default `400`)
  - `--silero-speech-pad-ms` (default `80`)

## Troubleshooting

- **`Missing required binary on PATH: ivector-subtract-global-mean`**
  - You forgot to source `path.sh`, or `KALDI_ROOT` is wrong.
  - Fix:

```bash
cd /home/shua/home/Shua/recipies/Diar/ssl_diarization
. ./path.sh
which ivector-subtract-global-mean
```

- **Silero VAD tries to download**
  - First run uses `torch.hub` and caches in `~/.cache/torch/hub/`.
  - If you’re offline, run once on a machine with network (or pre-populate the cache).

- **Checkpoint shape mismatch**
  - Pass the correct `--channels` and `--fc-dim` to match the checkpoint.

## Repo layout (high level)

- **`run_barlow_plda_cluster.py`**: simple single-wav diarization pipeline (recommended entry point).
- **`barlow_model.py`**: 16 kHz Barlow model used for inference here.
- **`iter_67_plda/`**: example PLDA model artifacts (`mean.vec`, `plda_model`).
- **`path.sh`**: Kaldi environment setup for this repo.
- **`local/train_plda_on_neil_validate.py`**: research script showing PLDA scoring + clustering logic.
- **`libs/aggHC.py`**: clustering + RTTM utilities used by older scripts.

