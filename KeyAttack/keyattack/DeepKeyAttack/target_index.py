class TargetIndexing:
    """
    Takes in labels and assign indices as they arrive.

    Code by Kevin Zhang,
    """

    def __init__(self, prev_labels=None):
        self.labels = [] if prev_labels is None else prev_labels

    def get_index(self, target):
        """
        Gets index of a target, appends to list if not exists.
        """

        if target in self.labels:
            return self.labels.index(target)
        else:
            self.labels.append(target)
            return len(self.labels) - 1

    def get_target(self, ind):
        """
        Gets the target from the index.
        """

        return self.labels[ind]

    def __str__(self):
        return f"TargetIndexing({str(self.labels)})"

    def __len__(self):
        return len(self.labels)
