import glob
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile

from KeyAttack import data_info

# Change to select dataset to load
SOURCE_INDEX = 0


def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    """
    Isolates keystrokes in file.

    Code from https://github.com/JBFH-Dev/Keystroke-Datasets

    :return: list of keystrokes.
    """
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


def process_audio(audio_folder, labels, stroke_count, test_data_ratio=0.):
    """
    Processes audio and saves into file.

    Code mostly from https://github.com/JBFH-Dev/Keystroke-Datasets,
    file-saving and splitting test data written by Kevin Zhang.

    :param audio_folder: folder containing the audio files
    :param labels: all labels for the data
    :param test_data_ratio: amount of test data
    """
    if not 0 <= test_data_ratio <= 1:
        raise ValueError("Test data percentage needs to be between 0 and 1!")

    keys = [k + ".wav" for k in labels]
    data_dict = {"Key": [], "File": []}

    if test_data_ratio:
        data_dict["TestIndices"] = {}

    label_count = len(keys)

    for i, file in enumerate(keys):
        print("Progress: ", (i + 1) * 100 // label_count, "% file = ", file, sep="")

        # Load data
        loc = os.path.join(audio_folder, file)
        samples, sample_rate = librosa.load(loc, sr=None)
        # samples = samples[round(1*sample_rate):]

        # Find appropriate split for data
        strokes = []
        prom = 0.06
        step = 0.005
        while not len(strokes) == stroke_count:
            strokes = isolator(samples[1 * sample_rate:], sample_rate, 48, 24,
                               2400, 12000, prom, False)
            if len(strokes) < stroke_count:
                prom -= step
            if len(strokes) > stroke_count:
                prom += step
            if prom <= 0:
                print('-- not possible for: ', file)
                break
            step = step * 0.99

        # Remove incompatible data
        idx = 0
        while idx < len(strokes):
            if strokes[idx].shape[1] != 14400:
                del strokes[idx]
            else:
                idx += 1

        label = [labels[i]] * len(strokes)
        data_dict["Key"] += label
        data_dict["File"] += strokes

    if test_data_ratio:
        # Count instances of each label
        label_count = Counter(data_dict["Key"])
        for label, count in label_count.items():
            data_dict["TestIndices"][label] \
                = random.sample(range(count),
                                math.floor(count * test_data_ratio))

    # Create empty folders to put resulting data
    processed_folder = os.path.join(audio_folder, "processed")
    if test_data_ratio < 1:
        processed_folder = empty_folder(processed_folder)
    else:
        shutil.rmtree(processed_folder)

    test_processed_folder = os.path.join(audio_folder, "test_processed")
    if test_data_ratio:
        test_processed_folder = empty_folder(test_processed_folder)
    else:
        shutil.rmtree(test_processed_folder)

    # Save data in either processed or test_processed folder
    label_count = {}
    for label, stroke in zip(data_dict["Key"], data_dict["File"]):
        if label in label_count:
            index = label_count[label]
            label_count[label] += 1
        else:
            index = 0
            label_count[label] = 1

        if test_data_ratio and index in data_dict["TestIndices"][label]:
            output_folder = test_processed_folder
        else:
            output_folder = processed_folder

        wavfile.write(
            f"{output_folder}/{label}_{index}.wav",
            sample_rate,
            stroke[0].numpy()
        )


def empty_folder(path):
    """
    Creates a folder if it does not exist or remove all files if it does.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)

    return path


if __name__ == '__main__':
    process_audio(
        data_info.DATA_PATHS[SOURCE_INDEX],
        data_info.DATA_LABELS[SOURCE_INDEX],
        data_info.STROKE_COUNTS[SOURCE_INDEX],
        test_data_ratio=0.1)
