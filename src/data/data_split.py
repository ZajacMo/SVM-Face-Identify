"""Dataset splitting and synthetic sample generation.

The original project trains on CASIA-FaceV5, a dataset that cannot be
redistributed and requires a signed licence agreement. To keep this repository
usable without any download, :func:`generate_synthetic_dataset` creates a small
corpus of deterministic synthetic "face-like" images that exercises the exact
same code path as the real dataset.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence

__all__ = [
    "generate_synthetic_dataset",
    "split_dataset",
    "split",
    "load_label_file",
]


def _label_sort_key(label: str):
    """Sort labels numerically when possible and lexicographically otherwise.

    CASIA-FaceV5 uses zero-padded numeric folder names, which sort correctly
    as strings, but a user-supplied dataset may use names such as ``alice``.
    Falling back to a string comparison keeps such datasets working instead of
    raising ``ValueError``.

    Args:
        label: Class label to build a sort key for.

    Returns:
        A tuple that orders numeric labels before non-numeric ones and sorts
        each group naturally.
    """
    try:
        return (0, int(label), "")
    except (TypeError, ValueError):
        return (1, 0, str(label))


def split_dataset(
    source_dir: str,
    label_dir: str,
    test_size_per_class: int = 1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Split a directory of per-class images into train and test manifests.

    The source directory is expected to contain one sub-directory per class,
    each holding the images of that class::

        source_dir/
            000/000_0.bmp ...
            001/001_0.bmp ...

    Args:
        source_dir: Root directory containing one sub-directory per class.
        label_dir: Directory in which the JSON manifests are written.
        test_size_per_class: Number of images reserved for the test set per
            class. Classes with fewer images contribute all of their images to
            the training set.
        seed: Seed for the random selection, so repeated runs produce an
            identical split.

    Returns:
        A dictionary with the keys ``"classes"``, ``"train"`` and ``"test"``
        holding the in-memory manifests that were written to disk.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"dataset directory not found: {source_dir}")

    os.makedirs(label_dir, exist_ok=True)
    rng = random.Random(seed)

    train_labels: List[Dict[str, str]] = []
    test_labels: List[Dict[str, str]] = []
    class_names: List[str] = []

    for person in sorted(os.listdir(source_dir)):
        person_path = os.path.join(source_dir, person)
        if not os.path.isdir(person_path):
            continue

        images = sorted(f for f in os.listdir(person_path) if f.lower().endswith(".bmp"))
        images += sorted(
            f
            for f in os.listdir(person_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and f not in images
        )
        if not images:
            continue

        if len(images) <= test_size_per_class:
            test_images: Sequence[str] = []
            train_images = images
        else:
            test_images = rng.sample(images, test_size_per_class)
            # Set membership avoids the O(n^2) list scan of the original code
            # and cannot drop a duplicate filename twice.
            test_set = set(test_images)
            train_images = [img for img in images if img not in test_set]

        class_names.append(person)
        for img in train_images:
            train_labels.append(
                {"path": os.path.join(person_path, img), "label": person}
            )
        for img in test_images:
            test_labels.append(
                {"path": os.path.join(person_path, img), "label": person}
            )

    class_names.sort(key=_label_sort_key)

    with open(os.path.join(label_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    with open(os.path.join(label_dir, "train_labels.json"), "w", encoding="utf-8") as f:
        json.dump(train_labels, f, indent=2, ensure_ascii=False)
    with open(os.path.join(label_dir, "test_labels.json"), "w", encoding="utf-8") as f:
        json.dump(test_labels, f, indent=2, ensure_ascii=False)

    return {"classes": class_names, "train": train_labels, "test": test_labels}


def load_label_file(path: str) -> List[Dict[str, str]]:
    """Read a manifest produced by :func:`split_dataset`.

    Args:
        path: Path to the JSON manifest.

    Returns:
        A list of ``{"path": ..., "label": ...}`` dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_synthetic_dataset(
    output_dir: str,
    num_classes: int = 12,
    train_per_class: int = 8,
    test_per_class: int = 2,
    image_size: int = 48,
    seed: int = 42,
) -> str:
    """Generate a deterministic synthetic face-like dataset.

    Each class is rendered as a coloured ellipse on a noisy background. The
    classes are distinguished by hue, which spreads them evenly around the
    colour wheel, so a classifier can separate them reliably. This is a
    plumbing test, not a benchmark: the synthetic images carry no facial
    information, and the accuracy they produce says nothing about face
    recognition.

    Args:
        output_dir: Directory to populate with one sub-directory per class.
        num_classes: Number of distinct classes to generate.
        train_per_class: Number of training images per class.
        test_per_class: Number of test images per class.
        image_size: Side length of the generated square images.
        seed: Seed controlling the noise and the per-class colour.

    Returns:
        The path of ``output_dir``.
    """
    import colorsys

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    for class_idx in range(num_classes):
        class_name = f"{class_idx:03d}"
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Spread the classes evenly around the hue circle and keep the
        # saturation and value high so the colours stay clearly distinct.
        hue = class_idx / max(num_classes, 1)
        face_color = tuple(
            int(channel * 255)
            for channel in colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        )

        for img_idx in range(train_per_class + test_per_class):
            img = Image.new("RGB", (image_size, image_size), (30, 30, 30))
            draw = ImageDraw.Draw(img)
            for _ in range(image_size * image_size // 12):
                x = rng.randrange(image_size)
                y = rng.randrange(image_size)
                shade = rng.randrange(20, 70)
                draw.point((x, y), fill=(shade, shade, shade))

            jitter = rng.uniform(-2, 2)
            margin = image_size * 0.22
            draw.ellipse(
                [
                    margin + jitter,
                    margin + jitter,
                    image_size - margin + jitter,
                    image_size - margin + jitter,
                ],
                fill=face_color,
            )
            eye_dx = image_size * 0.12
            eye_y = image_size * 0.42
            for sign in (-1, 1):
                cx = image_size / 2 + sign * eye_dx
                draw.ellipse(
                    [cx - 2, eye_y - 2, cx + 2, eye_y + 2],
                    fill=(20, 20, 20),
                )
            draw.line(
                [
                    image_size / 2 - 4,
                    image_size * 0.62,
                    image_size / 2 + 4,
                    image_size * 0.62,
                ],
                fill=(20, 20, 20),
                width=1,
            )
            img.save(os.path.join(class_dir, f"{class_name}_{img_idx}.png"))

    return output_dir


def split(
    source_dir: str = "64_CASIA-FaceV5",
    label_dir: str = "labels",
    test_size_per_class: int = 1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Split a dataset using the default layout and report the result.

    Args:
        source_dir: Root directory containing one sub-directory per class.
        label_dir: Directory in which the JSON manifests are written.
        test_size_per_class: Number of images reserved for the test set per
            class.
        seed: Seed for the random selection.

    Returns:
        The manifests written to ``label_dir``.
    """
    manifests = split_dataset(source_dir, label_dir, test_size_per_class, seed)
    print(
        f"Wrote {len(manifests['train'])} train and {len(manifests['test'])} test "
        f"manifests for {len(manifests['classes'])} classes to {label_dir}/"
    )
    return manifests
