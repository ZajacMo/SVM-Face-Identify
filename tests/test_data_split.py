"""Tests for the dataset splitting and synthetic generation utilities."""

from __future__ import annotations

import json
import os

from src.data.data_split import generate_synthetic_dataset, split_dataset


def test_split_is_reproducible(synthetic_root, tmp_path):
    """The same seed must produce identical manifests."""
    first = split_dataset(str(synthetic_root), str(tmp_path / "a"), seed=5)
    second = split_dataset(str(synthetic_root), str(tmp_path / "b"), seed=5)
    assert first["train"] == second["train"]
    assert first["test"] == second["test"]


def test_different_seeds_differ(synthetic_root, tmp_path):
    """A different seed must change the held-out images."""
    first = split_dataset(str(synthetic_root), str(tmp_path / "a"), seed=1)
    second = split_dataset(str(synthetic_root), str(tmp_path / "b"), seed=2)
    assert first["test"] != second["test"]


def test_split_writes_manifests(synthetic_root, tmp_path):
    """All three manifest files must be written and parse as JSON."""
    label_dir = tmp_path / "labels"
    split_dataset(str(synthetic_root), str(label_dir), test_size_per_class=2, seed=3)
    for name in ("classes.json", "train_labels.json", "test_labels.json"):
        path = label_dir / name
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            assert json.load(f) is not None


def test_split_partitions_every_image(synthetic_root, tmp_path):
    """Every image must land in exactly one of the two splits."""
    label_dir = tmp_path / "labels"
    manifests = split_dataset(
        str(synthetic_root), str(label_dir), test_size_per_class=2, seed=3
    )
    train_paths = {entry["path"] for entry in manifests["train"]}
    test_paths = {entry["path"] for entry in manifests["test"]}
    assert not train_paths & test_paths

    on_disk = set()
    for class_dir in os.listdir(str(synthetic_root)):
        class_path = os.path.join(str(synthetic_root), class_dir)
        if os.path.isdir(class_path):
            for name in os.listdir(class_path):
                on_disk.add(os.path.join(class_path, name))
    assert train_paths | test_paths == on_disk


def test_split_supports_non_numeric_labels(tmp_path):
    """Class names that are not integers must not raise."""
    root = tmp_path / "data"
    for name in ("alice", "bob"):
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for i in range(3):
            (class_dir / f"{i}.png").write_bytes(b"x")
    manifests = split_dataset(str(root), str(tmp_path / "labels"), seed=1)
    assert sorted(manifests["classes"]) == ["alice", "bob"]


def test_split_rejects_missing_directory(tmp_path):
    """A missing source directory must raise a clear error."""
    import pytest

    with pytest.raises(FileNotFoundError):
        split_dataset(str(tmp_path / "nope"), str(tmp_path / "labels"))


def test_generate_synthetic_dataset_layout(tmp_path):
    """The generator must create one directory per class with the images."""
    root = tmp_path / "synth"
    generate_synthetic_dataset(str(root), num_classes=3, train_per_class=2, test_per_class=1)
    classes = sorted(os.listdir(str(root)))
    assert classes == ["000", "001", "002"]
    for class_name in classes:
        images = os.listdir(str(root / class_name))
        assert len(images) == 3
