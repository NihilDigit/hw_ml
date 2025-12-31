from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def binary_rates_from_multiclass(y_true, y_pred, benign_label="Benign"):
    y_true_bin = np.array([0 if y == benign_label else 1 for y in y_true])
    y_pred_bin = np.array([0 if y == benign_label else 1 for y in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr


def overall_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
