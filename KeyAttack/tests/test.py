import glob, os, random
import json
import os
from collections import defaultdict
from infer import predict, TEST_AUDIO_DATA_INDEX
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
    with open(file_path, 's') as file:
        data = json.load(file)
    return data

def get_absolute_paths(directory):
    return [str(Path(directory) / file) for file in os.listdir(directory)]

def evaluate_predictions(predictions, ground_truths):
    correct_predictions = sum(pred == truth for pred, truth in zip(predictions, ground_truths))
    accuracy = correct_predictions / len(ground_truths)
    return accuracy

if __name__ == "__main__":
    demo_text_path = "demo_text.txt"
    demo_audio_directory = "path/demo/audio/"
    demo_data = load_demo_data(demo_text_path)
    demo_audio_paths = get_absolute_paths(demo_audio_directory)
    TEST_AUDIO_DATA_INDEX = 3
    predictions = predict(demo_audio_paths)
    ground_truths = []
    for string, labels in demo_data.items():
        ground_truths.extend(labels)
    accuracy = evaluate_predictions(predictions, ground_truths)
    print(f"Accuracy: {accuracy * 100:.2f}%")
