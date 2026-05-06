"""Classification metrics for OmniCT.

We report metrics chosen for class-imbalanced medical classification
(per Section 4.3 of the proposal):

- ROC-AUC          : ranking-based, robust to class imbalance.
- F1 (binary)      : harmonic mean of precision/recall.
- Balanced accuracy: mean of TPR and TNR.
- Sensitivity / Specificity (reported alongside).
- Brier score      : a calibration sanity check.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


METRIC_NAMES = (
    "roc_auc",
    "f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "brier",
    "accuracy",
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute headline metrics for binary classification.

    Args:
        y_true:  ground-truth labels in {0, 1}.
        y_score: predicted probability of the positive class.
        threshold: decision threshold for hard-label metrics.
    """
    y_true_a = np.asarray(list(y_true)).astype(int)
    y_score_a = np.asarray(list(y_score)).astype(float)
    y_pred_a = (y_score_a >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "roc_auc": _safe_auc(y_true_a, y_score_a),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_a, y_pred_a)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y_true_a, y_score_a)),
        "accuracy": float((y_pred_a == y_true_a).mean()),
    }
