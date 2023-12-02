class TargetIndexing:
    def __init__(self, prev_labels=None):
        self.labels = [] if prev_labels is None else prev_labels

    def get_index(self, target):
        if target in self.labels:
            return self.labels.index(target)
        else:
            self.labels.append(target)
            return len(self.labels) - 1

    def get_target(self, ind):
        return self.labels[ind]
