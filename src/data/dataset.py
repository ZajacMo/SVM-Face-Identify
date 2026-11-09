"""Shared dataset loading used by both the SVM and the neural network.

Both models previously built their own ``label -> index`` mapping inside the
loader. Because the mapping was derived from whichever manifest happened to be
read, the training set and the test set could end up with *different* index
assignments, which silently turns the reported accuracy into noise. This module
builds the mapping once, from the training manifest, and reuses it everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.utils import preprocess_image

__all__ = ["ClassMapping", "FaceDataset", "load_manifest", "build_class_mapping"]


@dataclass
class ClassMapping:
    """Bidirectional mapping between class labels and integer indices.

    Attributes:
        class_to_idx: Label to index mapping.
        idx_to_class: Index to label mapping, the inverse of ``class_to_idx``.
    """

    class_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_class: Dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of classes."""
        return len(self.class_to_idx)

    def encode(self, label: str) -> int:
        """Return the index of a label.

        Args:
            label: Class label to encode.

        Returns:
            The integer index of the label.

        Raises:
            KeyError: If the label is not part of the mapping.
        """
        return self.class_to_idx[label]

    def decode(self, index: int) -> str:
        """Return the label of an index.

        Args:
            index: Integer class index.

        Returns:
            The corresponding class label.
        """
        return self.idx_to_class[index]

    def one_hot(self, label: str) -> np.ndarray:
        """Return the one-hot encoded vector of a label.

        Args:
            label: Class label to encode.

        Returns:
            A float array of length ``len(self)`` with a single ``1.0``.
        """
        vector = np.zeros(len(self), dtype=np.float64)
        vector[self.encode(label)] = 1.0
        return vector


def _label_sort_key(label: str):
    """Sort numeric labels numerically and other labels lexicographically.

    Args:
        label: Class label to build a sort key for.

    Returns:
        A tuple ordering numeric labels first and non-numeric labels after.
    """
    try:
        return (0, int(label), "")
    except (TypeError, ValueError):
        return (1, 0, str(label))


def build_class_mapping(labels: Sequence[str]) -> ClassMapping:
    """Build a class mapping from an arbitrary sequence of labels.

    Args:
        labels: Label values observed in the training manifest.

    Returns:
        A :class:`ClassMapping` covering every distinct label.
    """
    unique = sorted(set(labels), key=_label_sort_key)
    class_to_idx = {label: i for i, label in enumerate(unique)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}
    return ClassMapping(class_to_idx=class_to_idx, idx_to_class=idx_to_class)


def load_manifest(path: str) -> List[Dict[str, str]]:
    """Read a JSON manifest of ``{"path", "label"}`` entries.

    Args:
        path: Path to the manifest.

    Returns:
        The list of manifest entries.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"label file not found: {path}") from exc


@dataclass
class FaceDataset:
    """A loaded image dataset with a fixed class mapping.

    Attributes:
        features: Feature matrix of shape ``(n_samples, n_features)``.
        labels: Integer label vector of shape ``(n_samples,)``.
        mapping: The class mapping shared with the training set.
    """

    features: np.ndarray
    labels: np.ndarray
    mapping: ClassMapping

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.features.shape[0])

    def one_hot_labels(self) -> np.ndarray:
        """Return the labels as a one-hot encoded matrix.

        Returns:
            A float array of shape ``(n_samples, n_classes)``.
        """
        matrix = np.zeros((len(self), len(self.mapping)), dtype=np.float64)
        matrix[np.arange(len(self)), self.labels] = 1.0
        return matrix

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        mapping: Optional[ClassMapping] = None,
        image_size: Tuple[int, int] = (32, 32),
        skip_missing: bool = True,
    ) -> "FaceDataset":
        """Load the images referenced by a manifest.

        Args:
            manifest_path: Path to the manifest to load.
            mapping: Class mapping to encode labels with. When omitted, a
                mapping is derived from the manifest itself, which is only
                appropriate for the training manifest. Labels absent from the
                mapping are skipped, which happens when the test set contains
                a class the training set never saw.
            image_size: Target ``(width, height)`` in pixels.
            skip_missing: Skip images that cannot be read instead of raising.

        Returns:
            The loaded dataset.
        """
        entries = load_manifest(manifest_path)
        if mapping is None:
            mapping = build_class_mapping([entry["label"] for entry in entries])
        features: List[np.ndarray] = []
        labels: List[int] = []
        for entry in entries:
            label = entry["label"]
            if label not in mapping.class_to_idx:
                continue
            try:
                features.append(preprocess_image(entry["path"], image_size))
            except (OSError, ValueError):
                if not skip_missing:
                    raise
                continue
            labels.append(mapping.encode(label))
        if not features:
            return cls(
                features=np.zeros((0, image_size[0] * image_size[1] * 3)),
                labels=np.zeros((0,), dtype=np.int64),
                mapping=mapping,
            )
        return cls(
            features=np.asarray(features, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int64),
            mapping=mapping,
        )
