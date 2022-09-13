'''
    File name: generalUtils.py
    Author: Shua Dissen
    Date created: 02/04/2021
    Date last modified: 02/04/2021
    Python Version: 3.8
'''

import os
import sys
import shutil
import argparse
from operator import attrgetter
import webrtcvad
import collections
import contextlib
import wave
import numpy as np

########################
def createDir(dir_name, clean_dir = False):
    if clean_dir:
      shutil.rmtree(dir_name, ignore_errors=True)   
    try:
      os.stat(dir_name)
    except:
      os.makedirs(dir_name, exist_ok=True)

########################
def spliceFiles(files, sampleRate=16000):
    return ""

########################
def concatinateWavs(infiles, outfile):
    with wave.open(outfile, 'wb') as wav_out:
        for wav_path in infiles:
            with wave.open(wav_path, 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                else:
                    assert(wav_out.getparams().nchannels == wav_in.getparams().nchannels)
                    assert(wav_out.getparams().sampwidth == wav_in.getparams().sampwidth)
                    assert(wav_out.getparams().framerate == wav_in.getparams().framerate)
                    
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

########################
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

########################
def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

########################
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

########################
def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0 
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n  
        # print(offset, timestamp)
########################
def frame_generator_test(frame_duration_ms, audio, sample_rate, timestamp=0.0):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0 + int(timestamp*sample_rate)*2
    timestamp = timestamp
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n       

########################
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames, log):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    times = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                log.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s) \n' % (frame.timestamp + frame.duration))
                log.write('-(%s)' % (frame.timestamp + frame.duration))
                log.write('\n')
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        log.write('-(%s)' % (frame.timestamp + frame.duration))
        log.write('\n')
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
      
########################
def runG_VAD_onFile(file, outwav="voicedAudio.wav", level=3, padding=300):
    vad = webrtcvad.Vad(level)
    audio, sample_rate = read_wave(file)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    log = open(outwav.replace('.wav', '.log'), 'w')
    segments = vad_collector(sample_rate, 30, padding, vad, frames, log)
    
    # return voicedAudio
    outdir = outwav.replace('.wav', '/')
    os.makedirs(os.path.dirname(outdir), exist_ok = True) 
    for i, segment in enumerate(segments):
        path = '%s/chunk-%002d.wav' % (outdir, i,)
        # print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)
    
    
    # lst = list(segments)
    # voicedAudio  = [val for sublist in lst for val in sublist]
    # # import ipdb; ipdb.set_trace()
    # write_wave(outwav, bytes(voicedAudio), sample_rate)
    
    ########################
def runG_VAD_onTestFile(file, rttm, outwav="voicedAudio.wav", level=3, padding=300):
    vad = webrtcvad.Vad(level)
    audio, sample_rate = read_wave(file)
    timestamp = float(rttm[0].split()[3])
    start = int(timestamp*sample_rate)
    endStamp = (float(rttm[-1].split()[3]) + float(rttm[-1].split()[4]))
    end = int(endStamp * sample_rate)
    audio = audio[:end*2]
    frames = frame_generator_test(30, audio, sample_rate, timestamp)
    # frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    # import ipdb; ipdb.set_trace()
    log = open(outwav.replace('.wav', '.log'), 'w')
    segments = vad_collector(sample_rate, 30, padding, vad, frames, log)
    
    # return voicedAudio
    outdir = outwav.replace('.wav', '/')
    os.makedirs(os.path.dirname(outdir), exist_ok = True) 
    for i, segment in enumerate(segments):
        path = '%s/chunk-%002d.wav' % (outdir, i,)
        # print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)
    
    
    # lst = list(segments)
    # voicedAudio  = [val for sublist in lst for val in sublist]
    # # import ipdb; ipdb.set_trace()
    # write_wave(outwav, bytes(voicedAudio), sample_rate)
    
def runG_VAD_onTestFileNorttm(file, outwav="voicedAudio.wav", level=3, padding=300):
    vad = webrtcvad.Vad(level)
    audio, sample_rate = read_wave(file)
    frames = frame_generator_test(30, audio, sample_rate)
    frames = list(frames)
    # import ipdb; ipdb.set_trace()
    log = open(outwav.replace('.wav', '.log'), 'w')
    segments = vad_collector(sample_rate, 30, padding, vad, frames, log)
    outdir = outwav.replace('.wav', '/')
    os.makedirs(os.path.dirname(outdir), exist_ok = True) 
    for i, segment in enumerate(segments):
        path = '%s/chunk-%002d.wav' % (outdir, i,)
        # print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)
        
    
    
    