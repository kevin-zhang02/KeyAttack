from pathlib import Path

import __init__

import glob
import json
import os
import random

from KeyAttack.keyattack.DeepKeyAttack.infer import predict
from KeyAttack.data_info import MODEL_PATHS, LABEL_COUNTS

TEST_AUDIO_DATA_INDEX = 2


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


def get_demo_audio_paths(demo_audio_folder):
    demo_audio_paths = [os.path.join(demo_audio_folder, filename) for filename in os.listdir(demo_audio_folder)]
    return demo_audio_paths


def compare_predictions_with_labels(predictions, ground_truths):
    correct_predictions = sum(pred == ground_truths[i] for i, pred in enumerate(predictions))
    accuracy = correct_predictions / len(predictions)
    return accuracy


def accuracy_compare():
    if not 2 <= TEST_AUDIO_DATA_INDEX <= 3:
        raise ValueError("Can only test this using "
                         "Curtis's (2) or Nayan's (3) dataset")

    demo_audio_folder = os.path.abspath('demo/audio/')

    # process_audio(
    #     demo_audio_folder,
    #     [
    #         Path(file).stem for file in os.listdir("demo/audio")
    #         if os.path.isfile(os.path.join(demo_audio_folder, file))
    #     ],
    #     133
    # )

    demo_text = "demo/demo_text.txt"
    demo_data = load_demo_data(demo_text)
    demo_audio_paths = get_demo_audio_paths(
        "demo/audio/processed"
    )
    predictions = predict(
        demo_audio_paths,
        MODEL_PATHS[TEST_AUDIO_DATA_INDEX],
        LABEL_COUNTS[TEST_AUDIO_DATA_INDEX]
    )
    ground_truths = demo_data.values().copy()
    accuracy = compare_predictions_with_labels(predictions, ground_truths)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    accuracy_compare()
