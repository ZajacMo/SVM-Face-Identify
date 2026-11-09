"""Evaluation of the trained BP neural network on a held-out manifest."""

from __future__ import annotations

import datetime
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.dataset import ClassMapping, build_class_mapping
from src.testing.metrics import classification_metrics, format_report, save_metrics
from src.utils.utils import preprocess_image

__all__ = ["FaceRecognitionTester", "test"]


class FaceRecognitionTester:
    """Runs predictions with a persisted BP neural network.

    Attributes:
        model: The unpickled network.
        class_to_idx: Label to index mapping saved alongside the model.
        idx_to_class: Index to label mapping, the inverse of ``class_to_idx``.
    """

    def __init__(self, model_path: str, labels_path: str) -> None:
        """Load a model and its class mapping from disk.

        Args:
            model_path: Path to the pickled network.
            labels_path: Path to the pickled class mapping.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels file not found: {labels_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.class_to_idx = pickle.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def predict(
        self, image_path: str, image_size: Tuple[int, int] = (32, 32)
    ) -> Tuple[str, float]:
        """Predict the identity of a single image.

        Args:
            image_path: Path to the image.
            image_size: Target ``(width, height)`` in pixels.

        Returns:
            A pair ``(predicted_label, confidence)``.
        """
        img_array = preprocess_image(image_path, image_size).reshape(1, -1)
        output = self.model.forward(img_array)
        predicted_idx = int(np.argmax(output))
        confidence = float(output[0][predicted_idx])
        return self.idx_to_class[predicted_idx], confidence

    def automatic_test(
        self,
        test_labels_path: str,
        image_size: Tuple[int, int] = (32, 32),
        output_dir: str = "results/plots",
    ) -> Dict[str, object]:
        """Evaluate the model over every entry of a manifest.

        Missing images are skipped and reported, so a partially available
        dataset still produces a usable result instead of aborting.

        Args:
            test_labels_path: Path to the test manifest.
            image_size: Target ``(width, height)`` in pixels.
            output_dir: Directory in which the log and JSON results are
                written.

        Returns:
            A dictionary with the accuracy, the per-sample results and the
            classification report.
        """
        with open(test_labels_path, "r", encoding="utf-8") as f:
            test_samples = json.load(f)

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "test_log.txt")

        correct = 0
        total = len(test_samples)
        results: List[Dict[str, object]] = []
        missing: List[str] = []
        y_true: List[int] = []
        y_pred: List[int] = []

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== Automatic test mode ===\n")
            log_file.write(f"Test time: {timestamp}\n")
            log_file.write(f"Test manifest: {test_labels_path}\n")
            log_file.write(f"Total test samples: {total}\n\n")

            for sample in test_samples:
                img_path = sample["path"]
                actual_person = sample["label"]
                if not os.path.exists(img_path):
                    missing.append(img_path)
                    log_file.write(f"WARNING: image not found - {img_path}\n")
                    continue

                predicted_person, confidence = self.predict(img_path, image_size)
                is_correct = predicted_person == actual_person
                log_file.write(f"Image: {img_path}\n")
                log_file.write(f"Actual person: {actual_person}\n")
                log_file.write(
                    f"Predicted person: {predicted_person} "
                    f"(confidence: {confidence:.2f})\n"
                )
                log_file.write(f"Result: {'correct' if is_correct else 'wrong'}\n\n")

                results.append(
                    {
                        "image_path": img_path,
                        "actual_person": actual_person,
                        "predicted_person": predicted_person,
                        "confidence": confidence,
                        "result": "correct" if is_correct else "wrong",
                    }
                )
                if actual_person in self.class_to_idx:
                    y_true.append(self.class_to_idx[actual_person])
                    y_pred.append(self.class_to_idx[predicted_person])
                if is_correct:
                    correct += 1

            accuracy = correct / total if total else 0.0
            log_file.write("\n=== Summary ===\n")
            log_file.write(f"Test finished: {timestamp}\n")
            log_file.write(f"Total test samples: {total}\n")
            log_file.write(f"Correct predictions: {correct}\n")
            log_file.write(f"Missing images: {len(missing)}\n")
            log_file.write(f"Accuracy: {accuracy:.2f} ({correct}/{total})\n")

        metrics = classification_metrics(y_true, y_pred) if y_true else {
            "accuracy": accuracy,
            "n_samples": 0,
        }
        payload = {
            "accuracy": float(accuracy),
            "total_tests": total,
            "correct": correct,
            "missing": len(missing),
            "metrics": metrics,
            "classification_report": format_report(y_true, y_pred) if y_true else "",
            "test_cases": results,
            "timestamp": timestamp,
        }
        save_metrics(payload, os.path.join(output_dir, "test_results.json"))

        print(f"Automatic test finished; log written to {log_path}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        return payload


def test(
    model_path: str = "results/face_model.pkl",
    labels_path: str = "results/class_labels.pkl",
    test_labels_path: str = "labels/test_labels.json",
    image_size: Tuple[int, int] = (32, 32),
    output_dir: str = "results/plots",
) -> Dict[str, object]:
    """Evaluate a persisted model over a test manifest.

    Args:
        model_path: Path to the pickled network.
        labels_path: Path to the pickled class mapping.
        test_labels_path: Path to the test manifest.
        image_size: Target ``(width, height)`` in pixels.
        output_dir: Directory for the log and JSON results.

    Returns:
        The evaluation payload.

    Raises:
        FileNotFoundError: If the model or labels file is missing.
    """
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            "model files not found, run the training step before testing"
        )
    tester = FaceRecognitionTester(model_path, labels_path)
    return tester.automatic_test(test_labels_path, image_size, output_dir)
