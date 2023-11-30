class TargetIndexing:
    def __init__(self, prev_labels=None):
        self.labels = [] if prev_labels is None else prev_labels

    def get_index(self, char):
        if char in self.labels:
            return self.labels.index(char)
        else:
            self.labels += char
            return len(self.labels) - 1

    def get_target(self, ind):
        return self.labels[ind]
