"""Shared pytest fixtures.

The fixtures build a tiny synthetic dataset on disk so the whole suite runs in
seconds and never needs the CASIA-FaceV5 download.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_split import generate_synthetic_dataset, split_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Return the root of a generated synthetic dataset.

    Returns:
        Path to the directory holding one sub-directory per class.
    """
    root = tmp_path_factory.mktemp("synthetic_dataset")
    generate_synthetic_dataset(
        str(root), num_classes=4, train_per_class=5, test_per_class=2, image_size=16
    )
    return root


@pytest.fixture(scope="session")
def manifests(synthetic_root, tmp_path_factory):
    """Split the synthetic dataset and return the manifest paths.

    Returns:
        A dictionary with the ``root``, ``label_dir``, ``train`` and ``test``
        keys.
    """
    label_dir = tmp_path_factory.mktemp("labels")
    split_dataset(str(synthetic_root), str(label_dir), test_size_per_class=2, seed=7)
    return {
        "root": str(synthetic_root),
        "label_dir": str(label_dir),
        "train": os.path.join(str(label_dir), "train_labels.json"),
        "test": os.path.join(str(label_dir), "test_labels.json"),
    }
