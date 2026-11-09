"""Head-to-head comparison of the BP neural network and the SVM."""

from __future__ import annotations

import datetime
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score

from src.data.dataset import build_class_mapping
from src.testing.metrics import classification_metrics, format_report, save_metrics
from src.training.svm_classifier import SVMClassifier
from src.utils.utils import preprocess_image

__all__ = ["ModelComparer", "compare"]


class ModelComparer:
    """Loads both models and evaluates them on the same test manifest.

    Attributes:
        bp_model: The unpickled BP neural network, if loaded.
        svm_model: The :class:`SVMClassifier`, if loaded.
        test_data: Feature matrix of the test set.
        test_labels: Integer labels of the test set.
        class_to_idx: Label to index mapping shared by both models.
    """

    def __init__(self) -> None:
        """Create an empty comparer."""
        self.bp_model = None
        self.svm_model: Optional[SVMClassifier] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.class_to_idx: Dict[str, int] = {}

    def load_bp_model(
        self, model_path: str, labels_path: str
    ) -> "ModelComparer":
        """Load the BP network and its class mapping.

        Args:
            model_path: Path to the pickled network.
            labels_path: Path to the pickled class mapping.

        Returns:
            ``self``, so calls can be chained.
        """
        with open(model_path, "rb") as f:
            self.bp_model = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.class_to_idx = pickle.load(f)
        return self

    def load_svm_model(self, model_path: str) -> "ModelComparer":
        """Load the SVM model.

        Args:
            model_path: Path to the pickled SVM.

        Returns:
            ``self``, so calls can be chained.
        """
        self.svm_model = SVMClassifier.load_model(model_path)
        return self

    def load_test_data(
        self,
        test_labels_path: str,
        image_size: Sequence[int] = (32, 32),
    ) -> "ModelComparer":
        """Load the test manifest into feature and label arrays.

        Args:
            test_labels_path: Path to the test manifest.
            image_size: Target ``(width, height)`` in pixels.

        Returns:
            ``self``, so calls can be chained.
        """
        with open(test_labels_path, "r", encoding="utf-8") as f:
            test_samples = json.load(f)

        features: List[np.ndarray] = []
        labels: List[int] = []
        for sample in test_samples:
            img_path = sample["path"]
            if not os.path.exists(img_path):
                continue
            features.append(preprocess_image(img_path, tuple(image_size)))
            if sample["label"] in self.class_to_idx:
                labels.append(self.class_to_idx[sample["label"]])

        self.test_data = np.asarray(features, dtype=np.float64)
        self.test_labels = np.asarray(labels, dtype=np.int64)
        return self

    def evaluate_bp(self) -> Dict[str, object]:
        """Evaluate the BP neural network.

        Returns:
            A dictionary with the accuracy, the metrics and the predictions.

        Raises:
            ValueError: If the model or the test data is missing.
        """
        if self.bp_model is None or self.test_data is None:
            raise ValueError("BP model or test data has not been loaded")

        predictions = [
            int(np.argmax(self.bp_model.forward(img.reshape(1, -1))[0]))
            for img in self.test_data
        ]
        metrics = classification_metrics(self.test_labels, predictions)
        return {
            "accuracy": float(
                accuracy_score(self.test_labels, np.asarray(predictions))
            ),
            "metrics": metrics,
            "predictions": predictions,
        }

    def evaluate_svm(self) -> Dict[str, object]:
        """Evaluate the SVM.

        Returns:
            A dictionary with the accuracy, the metrics and the predictions.

        Raises:
            ValueError: If the model or the test data is missing.
        """
        if self.svm_model is None or self.test_data is None:
            raise ValueError("SVM model or test data has not been loaded")

        predictions = self.svm_model.predict(self.test_data)
        metrics = classification_metrics(self.test_labels, predictions)
        return {
            "accuracy": float(accuracy_score(self.test_labels, predictions)),
            "metrics": metrics,
            "predictions": predictions.tolist(),
        }

    def compare(self, output_dir: str = "results/comparison") -> Dict[str, object]:
        """Evaluate both models and write the comparison artefacts.

        Args:
            output_dir: Directory for the JSON results and the bar chart.

        Returns:
            A dictionary with the accuracy of both models.
        """
        os.makedirs(output_dir, exist_ok=True)

        bp_results = self.evaluate_bp()
        svm_results = self.evaluate_svm()

        comparison = {
            "bp_accuracy": bp_results["accuracy"],
            "svm_accuracy": svm_results["accuracy"],
            "bp_metrics": bp_results["metrics"],
            "svm_metrics": svm_results["metrics"],
            "test_samples": int(len(self.test_labels)),
            "comparison_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_metrics(comparison, os.path.join(output_dir, "comparison_results.json"))

        self._plot_comparison(bp_results["accuracy"], svm_results["accuracy"], output_dir)

        print("Model comparison finished.")
        print(f"BP neural network accuracy: {bp_results['accuracy']:.4f}")
        print(f"SVM accuracy: {svm_results['accuracy']:.4f}")
        print(f"Results written to {output_dir}")
        return comparison

    @staticmethod
    def _plot_comparison(bp_accuracy: float, svm_accuracy: float, output_dir: str) -> None:
        """Render the accuracy bar chart.

        Args:
            bp_accuracy: Accuracy of the BP neural network.
            svm_accuracy: Accuracy of the SVM.
            output_dir: Directory in which the PNG is written.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 6))
        models = ["BP neural network", "SVM"]
        accuracies = [bp_accuracy, svm_accuracy]
        bars = axis.bar(models, accuracies, color=["blue", "orange"])
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Accuracy")
        axis.set_title("Classification accuracy: BP neural network vs SVM")
        for bar, value in zip(bars, accuracies):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.4f}",
                ha="center",
            )
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
        plt.close(figure)


def compare(
    bp_model_path: str = "results/face_model.pkl",
    labels_path: str = "results/class_labels.pkl",
    svm_model_path: str = "results/svm_face_model.pkl",
    test_labels_path: str = "labels/test_labels.json",
    output_dir: str = "results/comparison",
) -> Dict[str, object]:
    """Compare the two models on a test manifest.

    Args:
        bp_model_path: Path to the pickled BP network.
        labels_path: Path to the pickled class mapping.
        svm_model_path: Path to the pickled SVM.
        test_labels_path: Path to the test manifest.
        output_dir: Directory for the comparison artefacts.

    Returns:
        The comparison dictionary.
    """
    comparer = ModelComparer()
    comparer.load_bp_model(bp_model_path, labels_path)
    comparer.load_svm_model(svm_model_path)
    comparer.load_test_data(test_labels_path)
    return comparer.compare(output_dir)
