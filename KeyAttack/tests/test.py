import __init__

import glob
import json
import os
import random

from KeyAttack.keyattack.DeepKeyAttack.infer import predict
from KeyAttack.keyattack.DeepKeyAttack.train import MODEL_PATHS, LABEL_COUNTS
from pathlib import Path


def select_test_sequence(folder_path: str, labels: list[str]) -> list[str]:
    """
    Selects files from test folder to act as sequence.

    Written by Edward Ng.

    :param folder_path: test folder
    :param labels: sequence of keys to test
    :return: list of filenames corresponding to label sequence
    """

    sequence = [None] * len(labels)
    for index, label in enumerate(labels):
        files = glob.glob(f"{folder_path}\\{label}.wav")
        file = random.choice(files)
        sequence[index] = file
    
    return sequence


def eval(sequence: list[str], labels: list[str]) -> float:
    """
    Evaluates sequence and predicted labels.

    Written by Edward Ng.

    :param sequence: filepaths
    :param labels: labels
    :return: true positive rate.
    """
    true_positive = 0
    for seq_label, label in zip(sequence, labels):
        if os.path.basename(seq_label).split('_')[0] == label:
            true_positive += 1

    return true_positive / len(labels)


def load_demo_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_absolute_paths(directory):
    return [str(Path(directory) / file) for file in os.listdir(directory)]


def evaluate_predictions(predictions, ground_truths):
    correct_predictions = sum(pred == truth for pred, truth in zip(predictions, ground_truths))
    accuracy = correct_predictions / len(ground_truths)
    return accuracy


if __name__ == "__main__":
    demo_text = "demo/demo_text.txt"
    demo_data = load_demo_data(demo_text)
    demo_audio_paths = get_absolute_paths(demo_text)
    TEST_AUDIO_DATA_INDEX = 3
    predictions = predict(
        demo_audio_paths,
        MODEL_PATHS[TEST_AUDIO_DATA_INDEX],
        LABEL_COUNTS[TEST_AUDIO_DATA_INDEX]
    )
    ground_truths = demo_data.values().copy()
    accuracy = evaluate_predictions(predictions, ground_truths)
    print(f"Accuracy: {accuracy * 100:.2f}%")
