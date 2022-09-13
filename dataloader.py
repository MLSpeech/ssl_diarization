import random
import os
# os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/'
import sys
import json
from pathlib import Path
from loguru import logger
# from librosa.filters import mel as librosa_mel_fn

import torch
import torchaudio
import torch.nn.functional as F
torchaudio.set_audio_backend("sox_io")
from glob import glob


def find_audio_files(path, exts=[".wav"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(audio_path, audio_length),...]
    """
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        if "._" in file:
            continue
        siginfo = torchaudio.info(file)
        length = siginfo.num_frames // siginfo.num_channels
        meta.append((file, length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta
    
def find_audio_files_libricss(path, exts=[".wav"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(audio_path, audio_length),...]
    """
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        if "._" in file:
            continue
        if "mix.wav" not in file:
            continue
        siginfo = torchaudio.info(file)
        length = siginfo.num_frames // siginfo.num_channels
        meta.append((file, length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

def find_audio_files_callHome(path, exts=[".wav"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(audio_path, audio_length),...]
    """
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        if "._" in file:
            continue
        if "chunk" not in file:
            continue
        siginfo = torchaudio.info(file)
        length = siginfo.num_frames // siginfo.num_channels
        meta.append((file, length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta
 
def sample_segment(audio, n_samples, rand, ret_idx=False):
    """
    samples a random segment of `n_samples` from `audio`.
    if audio is shorter than `n_samples` then the original audio is zero padded.
    audio - tensor of shape [1, T]
    n_samples - int, this will be the new length of audio
    ret_idx - if True then the start and end indices will be returned
    """
    start, end = 0, audio.shape[1]

    if audio.shape[1] > n_samples:
        diff = audio.shape[1] - n_samples
        if rand:
            start = random.randint(0, diff)           
        end = start + n_samples
        audio = audio[:, start:end]
    elif audio.shape[1] < n_samples:
        diff = n_samples - audio.shape[1]
        audio = F.pad(audio, (0, diff))

    if ret_idx:
        return audio, (start, end)
    return audio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, json_manifest, n_samples=None, min_duration=0, max_duration=float("inf"), rand=True):
        self.n_samples = n_samples
        # load list of files
        logger.info(f"loading from: {json_manifest}")
        self.files = json.load(open(json_manifest, "r"))
        logger.info(f"files in manifest: {len(self.files)}")
        # filter files that are incorrect duration
        self.files = list(filter(lambda x: min_duration <= x[1] <= max_duration, self.files))
        logger.info(f"files after duration filtering: {len(self.files)}")
        self.rand=rand
        print(rand, self.rand)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, length = self.files[i]
        audio, sr = torchaudio.load(path)

        if self.n_samples:
            audio = sample_segment(audio, self.n_samples, self.rand)

        return audio

class AudioDatasetEval(torch.utils.data.Dataset):
    def __init__(self, json_manifest, n_samples=None, min_duration=0, max_duration=float("inf"), rand=True):
        self.n_samples = n_samples
        # load list of files
        logger.info(f"loading from: {json_manifest}")
        self.files = json.load(open(json_manifest, "r"))
        logger.info(f"files in manifest: {len(self.files)}")
        # filter files that are incorrect duration
        self.files = list(filter(lambda x: min_duration <= x[1] <= max_duration, self.files))
        logger.info(f"files after duration filtering: {len(self.files)}")
        self.rand=rand
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, length = self.files[i]
        audio, sr = torchaudio.load(path)

        if self.n_samples:
            audio = sample_segment(audio, self.n_samples, self.rand)

        return audio, path


if __name__ == "__main__":
    meta = []
    outJson = sys.argv[1]
    for path in sys.argv[2:]:
        meta += find_audio_files(path)
    with open(outJson, "w") as outfile:
        json.dump(meta, outfile, indent=4)
