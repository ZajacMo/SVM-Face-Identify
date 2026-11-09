"""End-to-end tests of the command line pipeline.

These exercise the full run on the synthetic dataset, which is what a new user
without the CASIA download will do first.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_parser, main  # noqa: E402


def test_parser_exposes_documented_options():
    """The parser must accept every documented flag."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--synthetic",
            "--epochs",
            "5",
            "--seed",
            "1",
            "--tune",
            "--cv",
            "2",
            "--output-dir",
            "out",
            "--image-size",
            "16",
        ]
    )
    assert args.synthetic is True
    assert args.epochs == 5
    assert args.tune is True
    assert args.cv == 2
    assert args.output_dir == "out"
    assert args.image_size == 16


def test_pipeline_runs_end_to_end(tmp_path):
    """The synthetic pipeline must complete and write its artefacts."""
    output_dir = tmp_path / "results"
    label_dir = tmp_path / "labels"

    exit_code = main(
        [
            "--synthetic",
            "--synthetic-classes",
            "4",
            "--epochs",
            "10",
            "--image-size",
            "16",
            "--cv",
            "2",
            "--label-dir",
            str(label_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    assert (output_dir / "face_model.pkl").exists()
    assert (output_dir / "svm_face_model.pkl").exists()
    assert (output_dir / "class_labels.pkl").exists()
    assert (output_dir / "training_history.pkl").exists()
    assert (output_dir / "summary.json").exists()
    assert (label_dir / "train_labels.json").exists()
    assert (label_dir / "test_labels.json").exists()


def test_pipeline_reports_metrics(tmp_path):
    """The summary must contain the measured SVM accuracy."""
    output_dir = tmp_path / "results"
    exit_code = main(
        [
            "--synthetic",
            "--synthetic-classes",
            "4",
            "--epochs",
            "10",
            "--image-size",
            "16",
            "--cv",
            "2",
            "--label-dir",
            str(tmp_path / "labels"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    with open(output_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["synthetic"] is True
    assert 0.0 <= summary["svm"]["test_accuracy"] <= 1.0
    assert summary["svm"]["cv_std"] >= 0.0


def test_invalid_step_is_rejected(tmp_path):
    """An unknown step name must abort with a parser error."""
    with pytest.raises(SystemExit):
        main(["--steps", "split,not_a_step", "--output-dir", str(tmp_path)])


def test_comparison_artefacts_are_written(tmp_path):
    """The comparison step must write the JSON results."""
    output_dir = tmp_path / "results"
    label_dir = tmp_path / "labels"
    main(
        [
            "--synthetic",
            "--synthetic-classes",
            "4",
            "--epochs",
            "10",
            "--image-size",
            "16",
            "--cv",
            "2",
            "--label-dir",
            str(label_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    comparison = output_dir / "comparison" / "comparison_results.json"
    assert comparison.exists()
    with open(comparison, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "bp_accuracy" in payload and "svm_accuracy" in payload
