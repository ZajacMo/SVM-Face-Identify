"""Command line entry point for the face recognition pipeline.

Examples:
    Run the whole pipeline on the bundled synthetic dataset::

        python main.py --synthetic --epochs 50

    Run the whole pipeline on a local CASIA-FaceV5 checkout::

        python main.py --dataset 64_CASIA-FaceV5 --tune

    Split a dataset and evaluate an already trained model::

        python main.py --dataset 64_CASIA-FaceV5 --steps split,test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from src.data.data_split import generate_synthetic_dataset, split_dataset
from src.testing.compare_models import ModelComparer
from src.testing.metrics import classification_metrics, save_metrics
from src.testing.test import FaceRecognitionTester
from src.training.svm_classifier import SVMClassifier
from src.training.train import train as train_bp

__all__ = ["build_parser", "main"]

#: The steps the pipeline can execute, in the order they run.
ALL_STEPS = ("split", "train", "test", "svm", "compare", "plot")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Face recognition baseline comparing a from-scratch BP neural "
            "network with an RBF-kernel SVM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="64_CASIA-FaceV5",
        help="Root directory holding one sub-directory per class.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Generate a small synthetic dataset instead of reading --dataset. "
            "Requires no download and runs end to end in seconds."
        ),
    )
    parser.add_argument(
        "--synthetic-classes",
        type=int,
        default=12,
        help="Number of classes to generate in synthetic mode.",
    )
    parser.add_argument(
        "--label-dir",
        default="labels",
        help="Directory holding the train/test manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for models, metrics and figures.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Side length of the square images fed to the models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Training epochs for the BP neural network.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Step size of the gradient update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Samples per gradient step.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Units in the hidden layer of the neural network.",
    )
    parser.add_argument(
        "--test-size-per-class",
        type=int,
        default=1,
        help="Images reserved for the test set in each class.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for all random operations."
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of folds for cross-validation and hyper-parameter search.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a cross-validated grid search for the SVM hyper-parameters.",
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=(
            "Comma separated subset of steps to run. Valid steps: "
            + ", ".join(ALL_STEPS)
        ),
    )
    return parser


def _resolve_steps(raw: str) -> List[str]:
    """Parse and validate the ``--steps`` argument.

    Args:
        raw: Comma separated step names.

    Returns:
        The validated list of steps.

    Raises:
        ValueError: If an unknown step name is present.
    """
    steps = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise ValueError(
            f"unknown step(s): {', '.join(unknown)}; valid steps are "
            f"{', '.join(ALL_STEPS)}"
        )
    return steps


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the pipeline.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on success, ``1`` on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        steps = _resolve_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
        return 1

    image_size = (args.image_size, args.image_size)
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "plots")
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = args.dataset
    if args.synthetic:
        dataset_dir = os.path.join(output_dir, "synthetic_dataset")
        print(f"Generating synthetic dataset in {dataset_dir} ...")
        generate_synthetic_dataset(
            dataset_dir,
            num_classes=args.synthetic_classes,
            seed=args.seed,
        )

    train_manifest = os.path.join(args.label_dir, "train_labels.json")
    test_manifest = os.path.join(args.label_dir, "test_labels.json")

    if "split" in steps:
        print("=" * 30 + " dataset split " + "=" * 30)
        manifests = split_dataset(
            dataset_dir,
            args.label_dir,
            test_size_per_class=args.test_size_per_class,
            seed=args.seed,
        )
        print(
            f"{len(manifests['train'])} train and {len(manifests['test'])} test "
            f"samples across {len(manifests['classes'])} classes"
        )

    bp_model_path = os.path.join(output_dir, "face_model.pkl")
    labels_path = os.path.join(output_dir, "class_labels.pkl")
    history_path = os.path.join(output_dir, "training_history.pkl")
    svm_model_path = os.path.join(output_dir, "svm_face_model.pkl")

    if "train" in steps:
        print("=" * 30 + " BP neural network training " + "=" * 30)
        train_bp(
            data_dir=train_manifest,
            model_save_path=bp_model_path,
            labels_save_path=labels_path,
            history_save_path=history_path,
            image_size=image_size,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    svm_summary: dict = {}
    if "svm" in steps:
        print("=" * 30 + " SVM " + "=" * 30)
        svm_summary = _run_svm(
            train_manifest,
            test_manifest,
            svm_model_path,
            image_size,
            args.tune,
            args.cv,
            args.seed,
        )

    if "test" in steps:
        print("=" * 30 + " BP neural network evaluation " + "=" * 30)
        if not os.path.exists(bp_model_path):
            print("Model not found, skipping the evaluation step.")
        else:
            tester = FaceRecognitionTester(bp_model_path, labels_path)
            tester.automatic_test(test_manifest, image_size, plots_dir)

    if "compare" in steps:
        print("=" * 30 + " model comparison " + "=" * 30)
        if not (os.path.exists(bp_model_path) and os.path.exists(svm_model_path)):
            print("Both models are required for the comparison, skipping.")
        else:
            comparer = ModelComparer()
            comparer.load_bp_model(bp_model_path, labels_path)
            comparer.load_svm_model(svm_model_path)
            comparer.load_test_data(test_manifest, image_size)
            comparer.compare(comparison_dir)

    if "plot" in steps:
        print("=" * 30 + " figures " + "=" * 30)
        from src.plotting.plot_training_history import plot_all

        plot_all(history_path, os.path.join(plots_dir, "test_results.json"), plots_dir)

    print("=" * 30 + " summary " + "=" * 30)
    summary = {"dataset": dataset_dir, "synthetic": bool(args.synthetic)}
    if svm_summary:
        summary["svm"] = {
            "test_accuracy": svm_summary.get("test_accuracy"),
            "cv_mean": svm_summary.get("cv_mean"),
            "cv_std": svm_summary.get("cv_std"),
            "best_params": (svm_summary.get("tuning") or {}).get("best_params"),
        }
    save_metrics(summary, os.path.join(output_dir, "summary.json"))
    print(json.dumps(summary, indent=2))
    return 0


def _run_svm(
    train_manifest: str,
    test_manifest: str,
    model_path: str,
    image_size,
    tune: bool,
    cv: int,
    seed: int,
) -> dict:
    """Train and evaluate the SVM, reusing the training class mapping.

    Args:
        train_manifest: Path to the training manifest.
        test_manifest: Path to the test manifest.
        model_path: Where to write the pickled model.
        image_size: Target ``(width, height)`` in pixels.
        tune: Whether to run a cross-validated grid search.
        cv: Number of folds.
        seed: Seed controlling the fold split.

    Returns:
        A dictionary describing the run.
    """
    from src.data.dataset import build_class_mapping, load_manifest

    classifier = SVMClassifier()
    X_train, y_train = classifier.load_dataset(train_manifest, image_size)
    mapping = classifier.mapping
    X_test, y_test = classifier.load_dataset(
        test_manifest, image_size, mapping=mapping
    )

    tuning = None
    if tune and len(np.unique(y_train)) >= cv:
        tuning = classifier.tune(X_train, y_train, cv=cv, seed=seed)
        print(f"Best parameters: {tuning['best_params']}")
        print(f"Best CV score: {tuning['best_score']:.4f}")

    classifier.train(X_train, y_train)
    cv_result = classifier.cross_validate(X_train, y_train, cv=cv, seed=seed)
    train_accuracy = classifier.evaluate(X_train, y_train)
    test_accuracy = classifier.evaluate(X_test, y_test)

    classifier.save_model(model_path)
    metrics = classification_metrics(y_test, classifier.predict(X_test))
    save_metrics(
        {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
            "cv_folds": cv_result["cv_folds"],
            "tuning": tuning,
            "metrics": metrics,
        },
        os.path.join(os.path.dirname(model_path), "svm_metrics.json"),
    )

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        f"Cross-validated accuracy: {cv_result['cv_mean']:.4f} "
        f"+/- {cv_result['cv_std']:.4f}"
    )
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_mean": cv_result["cv_mean"],
        "cv_std": cv_result["cv_std"],
        "tuning": tuning,
    }


if __name__ == "__main__":
    sys.exit(main())
