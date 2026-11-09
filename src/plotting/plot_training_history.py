"""Rendering of the training and evaluation figures.

All figures are written with the non-interactive ``Agg`` backend so the module
can run on a headless CI machine.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

__all__ = [
    "load_training_history",
    "plot_accuracy",
    "plot_confidence_distribution",
    "plot_confusion_matrix_from_results",
    "plot_loss_curve",
    "plot_all",
]


def load_training_history(history_path: str = "results/training_history.pkl"):
    """Load a pickled training history.

    Args:
        history_path: Path to the pickled history.

    Returns:
        The history dictionary, or ``None`` when the file does not exist.
    """
    try:
        with open(history_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(
            f"Training history not found at {history_path}; "
            "run the training step first."
        )
        return None


def _prepare_ax():
    """Return a figure and axis with the non-interactive backend configured.

    Returns:
        A pair ``(figure, axis)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    return figure, axis


def plot_accuracy(
    accuracy: float,
    save_path: str = "results/plots/accuracy.png",
    title: str = "Model accuracy",
) -> None:
    """Render the overall accuracy as a single bar.

    Args:
        accuracy: Accuracy in ``[0, 1]``.
        save_path: Destination PNG path.
        title: Title of the figure.
    """
    import matplotlib.pyplot as plt

    figure, axis = _prepare_ax()
    axis.bar(["accuracy"], [accuracy], color="skyblue")
    axis.set_ylim(0, 1.0)
    axis.set_title(title)
    axis.set_ylabel("accuracy")
    axis.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha="center")
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def plot_confidence_distribution(
    confidences: List[float],
    save_path: str = "results/plots/confidence_distribution.png",
    title: str = "Prediction confidence distribution",
) -> None:
    """Render the distribution of prediction confidences.

    Args:
        confidences: Confidence values in ``[0, 1]``.
        save_path: Destination PNG path.
        title: Title of the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axis = _prepare_ax()
    sns.histplot(confidences, bins=10, kde=True, color="green", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("confidence")
    axis.set_ylabel("sample count")
    axis.grid(True, linestyle="--", alpha=0.7)
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix_from_results(
    results: Dict[str, object],
    save_path: str = "results/plots/confusion_matrix.png",
    max_classes: int = 20,
) -> None:
    """Render a confusion matrix from a test-results payload.

    Args:
        results: A payload with a ``test_cases`` list holding ``actual_person``
            and ``predicted_person`` entries.
        save_path: Destination PNG path.
        max_classes: Maximum number of classes to display, keeping the figure
            legible for large label spaces.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cases = results.get("test_cases", []) or []
    if not cases:
        print("No test cases available; skipping the confusion matrix.")
        return

    actual = [case["actual_person"] for case in cases]
    predicted = [case["predicted_person"] for case in cases]
    unique_classes = list(dict.fromkeys(actual + predicted))
    truncated = len(unique_classes) > max_classes
    if truncated:
        unique_classes = unique_classes[:max_classes]

    matrix = confusion_matrix(actual, predicted, labels=unique_classes)
    figure, axis = _prepare_ax()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_classes,
        yticklabels=unique_classes,
        ax=axis,
    )
    title = "Confusion matrix (first 20 classes)" if truncated else "Confusion matrix"
    axis.set_title(title)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("actual class")
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def plot_loss_curve(
    history: Optional[Dict[str, object]],
    save_path: str = "results/plots/loss_curve.png",
) -> None:
    """Render the training loss over epochs.

    Args:
        history: The training history dictionary.
        save_path: Destination PNG path.
    """
    import matplotlib.pyplot as plt

    if not history or "loss_history" not in history:
        print("Training history is incomplete; skipping the loss curve.")
        return

    epochs, losses = zip(*history["loss_history"])
    learning_rate = history.get("learning_rate", "unknown")
    total_epochs = history.get("epochs", len(epochs))
    batch_size = history.get("batch_size", "unknown")

    figure, axis = _prepare_ax()
    axis.plot(epochs, losses, "b-", linewidth=2)
    axis.set_title(
        f"Training loss (learning rate: {learning_rate}, "
        f"epochs: {total_epochs}, batch size: {batch_size})"
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy loss")
    axis.grid(True, linestyle="--", alpha=0.7)
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def plot_all(
    history_path: str = "results/training_history.pkl",
    test_results_path: str = "results/plots/test_results.json",
    output_dir: str = "results/plots",
) -> None:
    """Render every available figure.

    Args:
        history_path: Path to the pickled training history.
        test_results_path: Path to the JSON test results.
        output_dir: Directory in which the figures are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    history = load_training_history(history_path)
    if history:
        plot_loss_curve(history, os.path.join(output_dir, "loss_curve.png"))

    if os.path.exists(test_results_path):
        with open(test_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        plot_accuracy(
            results.get("accuracy", 0.0), os.path.join(output_dir, "accuracy.png")
        )
        confidences = [
            case.get("confidence", 0.0) for case in results.get("test_cases", [])
        ]
        if confidences:
            plot_confidence_distribution(
                confidences, os.path.join(output_dir, "confidence_distribution.png")
            )
        plot_confusion_matrix_from_results(
            results, os.path.join(output_dir, "confusion_matrix.png")
        )
    else:
        print(f"Test results not found at {test_results_path}; skipping those plots.")

    print(f"All figures written to {output_dir}")
