import json
import os

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose

from KeyAttack.keyattack.DeepKeyAttack.CoAtNet import \
    CoAtNet
from KeyAttack.keyattack.DeepKeyAttack.target_index import \
    TargetIndexing
from KeyAttack.keyattack.DeepKeyAttack.train import \
    MODEL_PATHS, ToMelSpectrogram, LABEL_COUNTS


# assuming model and transform functions are already defined
# and 'MODEL_PATH' contains the path to the trained model 

TEST_AUDIO_DIRS = [
    os.path.abspath('../Data/Keystroke-Datasets/MBPWavs/test_processed'),
    os.path.abspath('../Data/Keystroke-Datasets/Zoom/test_processed'),
    os.path.abspath('../Data/CurtisMBP/test_processed'),
    os.path.abspath('../Data/NayanMK/test_processed')
]

TEST_AUDIO_DATA_INDEX = 3


def load_audio_clip(audio_path):
    return librosa.load(audio_path, sr=None)[0]


class PredictDataset(torch.utils.data.Dataset):
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


def load_model(path):
    model = CoAtNet(LABEL_COUNTS[TEST_AUDIO_DATA_INDEX])  # should match the architecture of the trained model
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(audio_paths):
    model = load_model(MODEL_PATHS[TEST_AUDIO_DATA_INDEX])

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
    audio_dir = TEST_AUDIO_DIRS[TEST_AUDIO_DATA_INDEX]
    audio_dir_contents = os.listdir(audio_dir)
    audio_paths = [os.path.join(audio_dir, filename) for filename in audio_dir_contents]
    predictions = predict(audio_paths)

    with open(MODEL_PATHS[TEST_AUDIO_DATA_INDEX] + "LabelIndices", 'r') as f:
        target_indexing = TargetIndexing(json.load(f))

    predictions = [target_indexing.get_target(prediction) for prediction in predictions]

    print(predictions)

    predictions = np.array(predictions)
    true_labels = np.array([filename.split('_')[0] for filename in audio_dir_contents])
    print(np.count_nonzero(predictions == true_labels) / len(audio_dir_contents))


if __name__ == "__main__":
    main()
