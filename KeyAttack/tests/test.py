from pathlib import Path

import numpy as np

import __init__

import glob
import json
import os
import random

from KeyAttack.keyattack.Data.load_data import process_audio
from KeyAttack.keyattack.DeepKeyAttack.infer import predict
from KeyAttack.data_info import MODEL_PATHS, LABEL_COUNTS, DEMO_AUDIO_FILES, \
    DEMO_PATH, DEMO_AUDIO_PROCESSED, DEMO_AUDIO_FOLDER
from KeyAttack.keyattack.DeepKeyAttack.target_index import TargetIndexing

TEST_AUDIO_DATA_INDEX = 2


def select_test_sequence(test_folder: str, labels: list[str]) -> list[str]:
    """
    Selects files from test folder to act as sequence.

    Written by Edward Ng.

    :param test_folder: test folder
    :param labels: sequence of keys to test
    :return: list of filenames corresponding to label sequence
    """

    sequence = [None] * len(labels)
    for index, label in enumerate(labels):
        files = glob.glob(f"{test_folder}\\{label}.wav")
        file = random.choice(files)
        sequence[index] = file
    
    return sequence


def load_demo_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_demo_audio_paths(demo_audio_folder, demo_file_stem, stroke_count):
    demo_audio_paths = [
        [
            os.path.join(demo_audio_folder, f"{file_stem}_{index}.wav")
            for index in range(stroke_count)
        ] for file_stem in demo_file_stem
    ]
    return demo_audio_paths


def compare_predictions_with_labels(predictions, ground_truths):
    correct_predictions = sum(pred == ground_truths[i] for i, pred in enumerate(predictions))
    accuracy = correct_predictions / len(predictions)
    return accuracy


def test_demo_audio():
    if not 2 <= TEST_AUDIO_DATA_INDEX <= 3:
        raise ValueError("Can only test this using "
                         "Curtis's (2) or Nayan's (3) dataset")

    demo_text = os.path.join(DEMO_PATH, 'demo_text.txt')
    demo_data = load_demo_data(demo_text)

    demo_file_stems = [
        Path(file).stem for file in DEMO_AUDIO_FILES[TEST_AUDIO_DATA_INDEX]
    ]

    process_audio(
        DEMO_AUDIO_FOLDER,
        demo_file_stems,
        demo_data["stroke_count"]
    )

    with open(MODEL_PATHS[TEST_AUDIO_DATA_INDEX] + "LabelIndices", 'r') as f:
        target_indexing = TargetIndexing(json.load(f))

    label_indices = [
        target_indexing.get_index(label)
        for label in demo_data["labels"]
    ]

    # Paths for every audio file in demo/audio
    demo_audio_paths_all = get_demo_audio_paths(
        DEMO_AUDIO_PROCESSED,
        demo_file_stems,
        demo_data["stroke_count"]
    )

    for demo_audio_paths in demo_audio_paths_all:
        predictions = predict(
            demo_audio_paths,
            MODEL_PATHS[TEST_AUDIO_DATA_INDEX],
            LABEL_COUNTS[TEST_AUDIO_DATA_INDEX]
        )
        print(np.count_nonzero(np.array(predictions) == np.array(label_indices))
              / demo_data["stroke_count"])


if __name__ == "__main__":
    test_demo_audio()
