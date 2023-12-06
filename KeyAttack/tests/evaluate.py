import torch


def evaluator(y_test, y_pred, num_classes):
    class Evaluation:
        class ClassMetrics:
            def __init__(self, tp, tn, fp, fn):
                self.conf_mat = [[tp, fp], [fn, tn]]
                self.accuracy = (tp + tn) / (tp + tn + fp + fn)
                self.precision = tp / (tp + fp)
                self.recall = tp / (tp + fn)
                self.f1 = 2 * self.precision * self.recall / (
                            self.precision + self.recall)

            def __str__(self) -> str:
                return f"Confusion Matrix:\n\
                    tp = {self.conf_mat[0][0]}\tfp = {self.conf_mat[0][1]}\n\
                    fn = {self.conf_mat[1][0]}\ttn = {self.conf_mat[1][1]}\n\
                    Accuracy: {self.accuracy:.3f}\n\
                    Precision: {self.precision:.3f}\n\
                    Recall: {self.recall:.3f}\n\
                    F1: {self.f1:.3f}"

        def __init__(self, confusion_matrix):
            self.conf_mat = confusion_matrix

            self.total_samples = torch.sum(confusion_matrix)
            self.num_classes = confusion_matrix.shape[0]

            pred_cls_sums = torch.sum(confusion_matrix, dim=1)
            actual_cls_sums = torch.sum(confusion_matrix, dim=0)

            tps = [confusion_matrix[i][i] for i in range(self.num_classes)]
            fps = [pred_cls_sums[i] - confusion_matrix[i][i] for i in
                   range(self.num_classes)]
            fns = [actual_cls_sums[i] - confusion_matrix[i][i] for i in
                   range(self.num_classes)]
            tns = [self.total_samples - fps[i] - fns[i] - tps[i] for i in
                   range(self.num_classes)]

            self.class_metrics = [
                Evaluation.ClassMetrics(tps[i], tns[i], fps[i], fns[i]) for i
                in range(self.num_classes)]

        def __str__(self) -> str:
            conf_mat_str = '\n\n'.join('\t'.join(
                str(self.conf_mat[i][j].item()) for j in
                range(self.num_classes)) for i in range(self.num_classes))
            class_metrics_strings = '\n'.join(
                f'Class {i}:\n{str(self.class_metrics[i])}' for i in
                range(self.num_classes))
            return f"Confusion Matrix:\n{conf_mat_str}\n{class_metrics_strings}"

    conf_mat = torch.zeros(num_classes, num_classes)

    for actual_class, pred_class in zip(y_test, y_pred):
        conf_mat[pred_class][actual_class] += 1

    return Evaluation(conf_mat)
