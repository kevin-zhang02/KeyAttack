import glob, os, random

def select_test_sequence(folder_path: str, labels: list[str]) -> list[str]:
    sequence = []
    for label in labels:
        files = glob.glob(folder_path + '\\'+ label + '.wav')
        file = random.choice(files)
        sequence.append(file)
    
    return sequence

def eval(sequence: list[str], labels: list[str]) -> int:
    TP = 0
    for seq_label, label in zip(sequence, labels):
        if os.path.basename(seq_label).split('_')[0] == label:
            TP += 1

    return TP / len(labels)
