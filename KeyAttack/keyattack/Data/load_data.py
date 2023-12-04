import glob
import os
import random
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile


def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(signal, sr=sample_rate)
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    # norm = np.linalg.norm(energy)
    # energy = energy/norm
    # -- energy'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(energy)
    threshed = energy > threshold
    # -- peaks'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(threshed.astype(float))
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    # '-- isolating keystrokes'
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            strokes.append(torch.tensor(keystroke)[None, :])
            if show:
                plt.figure(figsize=(7, 2))
                librosa.display.waveshow(keystroke, sr=sample_rate)
            prev_end = timestamp+after
    return strokes


def process_audio(audio_folder, labels):
    keys = [k + ".wav" for k in labels]
    data_dict = {"Key": [], "File": [], "TestIndices": {}}

    label_count = len(keys)

    for i, File in enumerate(keys):
        print("Progress: ", (i + 1) * 100 // label_count, "% file = ", File, sep="")

        loc = audio_folder + File
        samples, sample_rate = librosa.load(loc, sr=None)
        # samples = samples[round(1*sample_rate):]
        strokes = []
        prom = 0.06
        step = 0.005
        stroke_count = STROKE_COUNTS[SOURCE_INDEX]
        while not len(strokes) == stroke_count:
            strokes = isolator(samples[1 * sample_rate:], sample_rate, 48, 24,
                               2400, 12000, prom, False)
            if len(strokes) < stroke_count:
                prom -= step
            if len(strokes) > stroke_count:
                prom += step
            if prom <= 0:
                print('-- not possible for: ', File)
                break
            step = step * 0.99

        idx = 0
        while idx < len(strokes):
            if strokes[idx].shape[1] != 14400:
                del strokes[idx]
            else:
                idx += 1

        label = [labels[i]] * len(strokes)
        data_dict["Key"] += label
        data_dict["File"] += strokes

    label_count = Counter(data_dict["Key"])
    for label, count in label_count.items():
        data_dict["TestIndices"][label] = random.sample(range(count), count // 10)

    empty_folder(f"{audio_folder}processed")
    empty_folder(f"{audio_folder}test_processed")

    label_count = {}
    for label, stroke in zip(data_dict["Key"], data_dict["File"]):
        if label in label_count:
            index = label_count[label]
            label_count[label] += 1
        else:
            index = 0
            label_count[label] = 1

        if index in data_dict["TestIndices"][label]:
            output_folder = f"{audio_folder}test_processed"
        else:
            output_folder = f"{audio_folder}processed"

        wavfile.write(f"{output_folder}/{label}_{index}.wav", sample_rate, stroke[0].numpy())


def empty_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    process_audio(DATA_PATHS[SOURCE_INDEX], DATA_LABELS[SOURCE_INDEX])
