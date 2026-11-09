"""Evaluation utilities shared by the neural network and the SVM.

Every metric is reported together with the number of samples it was computed
from, and cross-validated results are reported as a mean and a standard
deviation rather than as a single point estimate.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

__all__ = [
    "classification_metrics",
    "confusion_matrix_data",
    "save_metrics",
    "plot_confusion_matrix",
]


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Compute accuracy, macro precision/recall/F1 and a confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        labels: Optional explicit label ordering for the confusion matrix.

    Returns:
        A dictionary with the scalar metrics and the confusion matrix as a
        nested list.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "n_samples": int(y_true_arr.size),
        "confusion_matrix": matrix.tolist(),
    }


def confusion_matrix_data(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return the confusion matrix as an array.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        labels: Optional explicit label ordering.

    Returns:
        The confusion matrix.
    """
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)


def save_metrics(metrics: Dict[str, object], path: str) -> None:
    """Write metrics to a JSON file.

    Args:
        metrics: The metrics to serialise.
        path: Destination path. Parent directories are created.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Render a confusion matrix to a PNG file.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        labels: Label ordering used for the matrix axes.
        class_names: Human-readable names matching ``labels``.
        save_path: Destination PNG path. Parent directories are created.
        title: Title of the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix_data(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=list(class_names)
    )
    figure, _ = display.plot(cmap=plt.cm.Blues)
    figure.axes[0].set_title(title)
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def format_report(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    """Return the scikit-learn classification report as text.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        The rendered report.
    """
    return classification_report(np.asarray(y_true), np.asarray(y_pred), zero_division=0)


def summarize_cv(scores: List[float]) -> Dict[str, float]:
    """Summarise a list of cross-validation scores.

    Args:
        scores: Per-fold scores.

    Returns:
        A dictionary with the mean, the standard deviation and the fold count.
    """
    if not scores:
        return {"mean": 0.0, "std": 0.0, "folds": 0}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": len(scores),
    }
