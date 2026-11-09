"""Tests for the shared class mapping and dataset loader.

These cover the regression that motivated the module: the training and test
splits must agree on the label indices.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import FaceDataset, build_class_mapping


def test_mapping_covers_every_label():
    """Every distinct label must receive exactly one index."""
    mapping = build_class_mapping(["b", "a", "b"])
    assert mapping.class_to_idx == {"a": 0, "b": 1}
    assert len(mapping) == 2


def test_mapping_sorts_numeric_labels_numerically():
    """Numeric labels must sort as numbers, not as strings."""
    mapping = build_class_mapping(["10", "2", "1"])
    assert list(mapping.class_to_idx) == ["1", "2", "10"]


def test_mapping_handles_non_numeric_labels():
    """Non-numeric labels must not raise and must sort after numeric ones."""
    mapping = build_class_mapping(["2", "alice", "1"])
    assert list(mapping.class_to_idx) == ["1", "2", "alice"]


def test_mapping_one_hot_roundtrip():
    """The one-hot vector must select the encoded index."""
    mapping = build_class_mapping(["x", "y"])
    vector = mapping.one_hot("y")
    assert vector.tolist() == [0.0, 1.0]
    assert mapping.decode(mapping.encode("y")) == "y"


def test_dataset_shares_one_mapping_across_splits(manifests):
    """The train and test splits must use identical label indices."""
    mapping = build_class_mapping(
        [entry["label"] for entry in _load(manifests["train"])]
    )
    train = FaceDataset.from_manifest(manifests["train"], mapping, (16, 16))
    test = FaceDataset.from_manifest(manifests["test"], mapping, (16, 16))

    assert train.mapping is mapping
    assert test.mapping is mapping
    assert train.mapping.class_to_idx == test.mapping.class_to_idx

    # Every label present in the test manifest must be decodable with the
    # shared mapping.
    for label in {entry["label"] for entry in _load(manifests["test"])}:
        assert label in mapping.class_to_idx


def test_dataset_one_hot_shape(manifests):
    """The one-hot matrix must have one row per sample."""
    mapping = build_class_mapping(
        [entry["label"] for entry in _load(manifests["train"])]
    )
    train = FaceDataset.from_manifest(manifests["train"], mapping, (16, 16))
    one_hot = train.one_hot_labels()
    assert one_hot.shape == (len(train), len(mapping))
    assert np.all(one_hot.sum(axis=1) == 1.0)


def test_dataset_skips_missing_images(manifests, tmp_path):
    """Unreadable images must be skipped rather than aborting the load."""
    mapping = build_class_mapping(
        [entry["label"] for entry in _load(manifests["train"])]
    )
    broken = tmp_path / "broken.json"
    broken.write_text(
        '[{"path": "does/not/exist.png", "label": "000"}]', encoding="utf-8"
    )
    dataset = FaceDataset.from_manifest(str(broken), mapping, (16, 16))
    assert len(dataset) == 0


def _load(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
