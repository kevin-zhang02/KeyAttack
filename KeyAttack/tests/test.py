import glob, os, random

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

