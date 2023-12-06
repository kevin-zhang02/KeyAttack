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


def load_demo_data():
    demo_data_path = "demo_text.txt"
    with open(demo_data_path, 'r') as file:
        demo_data = json.load(file)
    return demo_data


def get_demo_audio_paths():
    demo_audio_folder = os.path.abspath('tests/demo/audio/')
    demo_audio_paths = [os.path.join(demo_audio_folder, filename) for filename in os.listdir(demo_audio_folder)]
    return demo_audio_paths


def compare_predictions_with_labels(predictions, ground_truths):
    correct_predictions = sum(pred == ground_truths[i] for i, pred in enumerate(predictions))
    accuracy = correct_predictions / len(predictions)
    return accuracy


def accuracy_compare():
    demo_data = load_demo_data()
    demo_audio_paths = get_demo_audio_paths()
    predictions = predict(demo_audio_paths)
    ground_truths = []
    for filename in demo_audio_paths:
        file_key = Path(filename).stem
        ground_truths.extend(demo_data.get(file_key, []))
    accuracy = compare_predictions_with_labels(predictions, ground_truths)
    print(f"Accuracy: {accuracy:.2%}")

