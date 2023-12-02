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
    keys = [k + '.wav' for k in labels]
    data_dict = {'Key': [], 'File': []}

    key_count = len(keys)
    progress = 0

    for i, File in enumerate(keys):
        if (i + 1) * 100 // key_count > progress:
            print("Progress: ", (i + 1) * 100 // key_count, "%", sep="")

        progress = (i + 1) * 100 // key_count

        loc = audio_folder + File
        samples, sample_rate = librosa.load(loc, sr=None)
        # samples = samples[round(1*sample_rate):]
        strokes = []
        prom = 0.06
        step = 0.005
        while not len(strokes) == 25:
            strokes = isolator(samples[1 * sample_rate:], sample_rate, 48, 24,
                               2400, 12000, prom, False)
            if len(strokes) < 25:
                prom -= step
            if len(strokes) > 25:
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
        data_dict['Key'] += label
        data_dict['File'] += strokes

    Path(f"{audio_folder}processed").mkdir(parents=True, exist_ok=True)

    key_count = {}
    for key, strokes in zip(data_dict["Key"], data_dict["File"]):
        for stroke in strokes:
            if key not in key_count:
                index = 0
                key_count[key] = 1
            else:
                index = key_count[key]
                key_count[key] = index + 1

            wavfile.write(f"{audio_folder}processed/{key}_{index}.wav", sample_rate, stroke.numpy())


if __name__ == '__main__':
    # process_audio('Keystroke-Datasets/MBPWavs/', '0123456789qwertyuiopasdfghjklzxcvbnm')
    # process_audio('CurtisMBP/', [
    #     *(str(i) for i in range(10)),
    #     *"abcdefghijklmnopqrstuvwxyz",
    #     "Backspace",
    #     "CapsLock",
    #     "Enter",
    #     "LeftShiftDown",
    #     "LeftShiftRelease",
    #     "-",
    #     ";",
    #     "[",
    #     "]",
    #     "=",
    #     "Apostrophe",
    #     "Backslash",
    #     "LeftAngleBracket",
    #     "RightAngleBracket",
    #     "Slash",
    #     "SpaceBar",
    #     "Tilde"
    # ])
    process_audio('NayanMK/', [
        *(str(i) for i in range(10)),
        *"abcdefghijklmnopqrstuvwxyz",
        "Backspace",
        "CapsLock",
        "Enter",
        "LeftShiftDown",
        "LeftShiftRelease",
        "-",
        ";",
        "[",
        "]",
        "=",
        "Apostrophe",
        "Backslash",
        "LeftAngleBracket",
        "RightAngleBracket",
        "Slash",
        "SpaceBar",
        "Tilde"
    ])
