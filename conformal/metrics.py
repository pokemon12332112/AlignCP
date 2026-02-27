
import numpy as np
from sklearn.metrics import confusion_matrix


def aca(output, target):

    cm = confusion_matrix(target, np.argmax(output, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))
    aca = np.round(np.mean(np.diag(cm_norm) * 100), 2)

    return aca


def accuracy(output, target, topk=(1,)):
    output, target = output.to("cpu"), target.to("cpu")
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_conformal(pred_sets, labels, alpha=0.1):

    size = set_size(pred_sets)
    coverage = empirical_set_coverage(pred_sets, labels)
    class_cov_gap = avg_class_coverage_gap(pred_sets, labels, alpha=alpha)

    return [coverage, size, class_cov_gap]


def set_size(pred_sets):
    mean_size = np.mean([len(pred_set) for pred_set in pred_sets])
    return mean_size


def empirical_set_coverage(pred_sets, labels):
    coverage = np.mean([label in pred_set for label, pred_set in zip(labels, pred_sets)])
    return coverage


def avg_class_coverage_gap(pred_sets, labels, alpha=0.1):
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    violation = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        violation.append(abs(correct[idx].mean() - (1 - alpha)))
    covgap = 100 * np.median(violation)

    return covgap


def avg_class_coverage(pred_sets, labels):
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    clas_cov = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        clas_cov.append(correct[idx].mean())
    clas_cov = np.mean(clas_cov)

    return clas_cov