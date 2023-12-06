import json
import os

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose

from KeyAttack.data_info import TEST_AUDIO_DIRS, MODEL_PATHS, LABEL_COUNTS
from KeyAttack.keyattack.DeepKeyAttack.CoAtNet import \
    CoAtNet
from KeyAttack.keyattack.DeepKeyAttack.target_index import \
    TargetIndexing
from KeyAttack.keyattack.DeepKeyAttack.train import ToMelSpectrogram


# assuming model and transform functions are already defined
# and 'MODEL_PATH' contains the path to the trained model

# Choose dataset to use
TEST_AUDIO_DATA_INDEX = 1


def load_audio_clip(audio_path):
    """
    Loads audio clip.
    """
    return librosa.load(audio_path, sr=None)[0]


class PredictDataset(torch.utils.data.Dataset):
    """
    Dataset class used for prediction.

    Code from https://github.com/soheil/DeepKeyAttack
    """

    def __init__(self, audio_paths, transform=None):
        self.audio_paths = audio_paths
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio_clip = load_audio_clip(audio_path)

        if self.transform:
            audio_clip = self.transform(audio_clip)

        return audio_clip


def load_model(path, label_count):
    """
    Loads model from path.

    Code ffrom https://github.com/soheil/DeepKeyAttack with minor changes by
    Kevin Zhang.
    """
    model = CoAtNet(label_count)  # should match the architecture of the trained model
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(audio_paths, model_path, label_count):
    """
    Predicts labels from data in audio_paths.

    Code from https://github.com/soheil/DeepKeyAttack

    :param audio_paths: list of .wav files
    :param model_path: path to the model
    :param label_count: the number of labels
    :return: predictions
    """
    model = load_model(model_path, label_count)

    transform = transforms.Compose([
        Compose([ToMelSpectrogram(), ToTensor()])
    ])

    dataset = PredictDataset(audio_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []

    for batch in data_loader:
        batch = batch
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)  # change if multi-label classification
        predictions.append(predicted.item())

    return predictions


def main():
    """
    Tests data found in test_processed folders.

    Code from https://github.com/soheil/DeepKeyAttack with changes to loading
    audio files, loading indices for labels, and prediction accuracy by
    Kevin Zhang.
    """
    audio_dir = TEST_AUDIO_DIRS[TEST_AUDIO_DATA_INDEX]
    audio_dir_contents = os.listdir(audio_dir)
    audio_paths = [os.path.join(audio_dir, filename) for filename in audio_dir_contents]
    predictions = predict(
        audio_paths,
        MODEL_PATHS[TEST_AUDIO_DATA_INDEX],
        LABEL_COUNTS[TEST_AUDIO_DATA_INDEX]
    )

    with open(MODEL_PATHS[TEST_AUDIO_DATA_INDEX] + "LabelIndices", 'r') as f:
        target_indexing = TargetIndexing(json.load(f))

    predictions = [target_indexing.get_target(prediction) for prediction in predictions]

    print(predictions)

    predictions = np.array(predictions)
    true_labels = np.array([filename.split('_')[0] for filename in audio_dir_contents])
    print(np.count_nonzero(predictions == true_labels) / len(audio_dir_contents))


if __name__ == "__main__":
    main()
