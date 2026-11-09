"""Training of the from-scratch BP neural network baseline."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.dataset import ClassMapping, FaceDataset, build_class_mapping
from src.utils.utils import BPNeuralNetwork

__all__ = ["load_casia_dataset", "train"]


def load_casia_dataset(
    label_file: str,
    image_size: Tuple[int, int] = (32, 32),
    mapping: Optional[ClassMapping] = None,
) -> Tuple[np.ndarray, np.ndarray, ClassMapping]:
    """Load a manifest into features and one-hot encoded targets.

    Args:
        label_file: Path to the JSON manifest.
        image_size: Target ``(width, height)`` in pixels.
        mapping: Class mapping to reuse. When omitted, a mapping is built from
            the manifest itself.

    Returns:
        A triple ``(features, one_hot_labels, mapping)``.
    """
    dataset = FaceDataset.from_manifest(label_file, mapping, image_size)
    return dataset.features, dataset.one_hot_labels(), dataset.mapping


def train(
    data_dir: str = "labels/train_labels.json",
    model_save_path: str = "results/face_model.pkl",
    labels_save_path: str = "results/class_labels.pkl",
    history_save_path: str = "results/training_history.pkl",
    image_size: Tuple[int, int] = (32, 32),
    hidden_size: int = 128,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, object]:
    """Train the BP neural network and persist the artefacts.

    Args:
        data_dir: Path to the training manifest.
        model_save_path: Where to write the pickled network.
        labels_save_path: Where to write the pickled class mapping.
        history_save_path: Where to write the pickled training history.
        image_size: Target ``(width, height)`` in pixels.
        hidden_size: Number of units in the hidden layer.
        epochs: Number of training epochs.
        learning_rate: Step size for the gradient update.
        batch_size: Number of samples per gradient step.
        seed: Seed applied before weight initialisation and shuffling.

    Returns:
        A dictionary describing the training run.
    """
    np.random.seed(seed)

    X, y_onehot, mapping = load_casia_dataset(data_dir, image_size)
    print(
        f"Dataset loaded: {X.shape[0]} images, {len(mapping)} classes "
        f"from {data_dir}"
    )

    input_size = image_size[0] * image_size[1] * 3
    output_size = len(mapping)
    model = BPNeuralNetwork(input_size, hidden_size, output_size, seed=seed)

    print("Starting model training...")
    model.train(
        X, y_onehot, epochs, learning_rate, batch_size, seed=seed
    )

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    with open(labels_save_path, "wb") as f:
        pickle.dump(mapping.class_to_idx, f)

    history = {
        "loss_history": model.loss_history,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    with open(history_save_path, "wb") as f:
        pickle.dump(history, f)

    print(f"Model saved to {model_save_path}")
    print(f"Class labels saved to {labels_save_path}")
    print(f"Training history saved to {history_save_path}")

    return {
        "train_samples": int(X.shape[0]),
        "num_classes": output_size,
        "final_loss": model.loss_history[-1][1] if model.loss_history else None,
        "loss_history": model.loss_history,
    }
