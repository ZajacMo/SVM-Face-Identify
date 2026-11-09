"""Support vector machine classifier for face recognition.

The SVM operates on raw pixel vectors, which makes it a useful, dependency-light
baseline rather than a competitive face recognition model. See the limitations
section of the README for a comparison with embedding-based approaches.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.data.dataset import ClassMapping, FaceDataset, build_class_mapping, load_manifest

__all__ = ["SVMClassifier", "svm_train", "DEFAULT_GRID"]

#: Candidate hyper-parameter grid searched when tuning is enabled.
DEFAULT_GRID: Dict[str, Sequence] = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
    "kernel": ["rbf"],
}


class SVMClassifier:
    """Wrapper around :class:`sklearn.svm.SVC` that owns its class mapping.

    Attributes:
        model: The fitted or unfitted scikit-learn estimator.
        class_to_idx: Label to index mapping, shared with the training set.
        idx_to_class: Index to label mapping, the inverse of ``class_to_idx``.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        probability: bool = True,
    ) -> None:
        """Create the classifier.

        Args:
            C: Regularisation parameter of the SVM.
            kernel: Kernel used by the SVM.
            gamma: Kernel coefficient; ``"scale"`` lets scikit-learn derive it
                from the feature variance.
            probability: Whether to enable probability estimates, which are
                required for :meth:`predict_proba`.
        """
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    @property
    def mapping(self) -> ClassMapping:
        """Return the class mapping as a :class:`ClassMapping`."""
        return ClassMapping(class_to_idx=self.class_to_idx, idx_to_class=self.idx_to_class)

    def load_dataset(
        self,
        label_file: str,
        image_size: Tuple[int, int] = (32, 32),
        mapping: Optional[ClassMapping] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a manifest into feature and label arrays.

        Args:
            label_file: Path to the JSON manifest.
            image_size: Target ``(width, height)`` in pixels.
            mapping: Class mapping to reuse. When omitted, a new mapping is
                built from this manifest, which is only appropriate for the
                training manifest. Pass the training mapping when loading the
                test set so both splits agree on the label indices.

        Returns:
            A pair ``(features, labels)`` of shape ``(n_samples, n_features)``
            and ``(n_samples,)``.
        """
        entries = load_manifest(label_file)
        if mapping is None:
            mapping = build_class_mapping([entry["label"] for entry in entries])
        self.class_to_idx = dict(mapping.class_to_idx)
        self.idx_to_class = dict(mapping.idx_to_class)
        dataset = FaceDataset.from_manifest(label_file, mapping, image_size)
        return dataset.features, dataset.labels

    def train(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Fit the underlying SVM.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer label vector of shape ``(n_samples,)``.

        Returns:
            ``self``, so calls can be chained.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted class indices.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy of the model on a dataset.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer label vector of shape ``(n_samples,)``.

        Returns:
            The fraction of correctly classified samples.
        """
        return float(accuracy_score(y, self.predict(X)))

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Evaluate the current hyper-parameters with stratified k-fold CV.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer label vector of shape ``(n_samples,)``.
            cv: Number of folds.
            seed: Seed for the fold split, so results are reproducible.

        Returns:
            A dictionary with the mean and standard deviation of the
            cross-validated accuracy.
        """
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = cross_val_score(
            self.model, X, y, cv=splitter, scoring="accuracy", n_jobs=-1
        )
        return {
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "cv_folds": [float(s) for s in scores],
        }

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, Sequence]] = None,
        cv: int = 5,
        seed: int = 42,
    ) -> Dict[str, object]:
        """Search for better hyper-parameters with cross-validated grid search.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer label vector of shape ``(n_samples,)``.
            param_grid: Grid to search. Defaults to :data:`DEFAULT_GRID`.
            cv: Number of folds.
            seed: Seed for the fold split.

        Returns:
            A dictionary with the ``best_params``, the ``best_score`` and the
            per-fold scores of the winning candidate.
        """
        grid = param_grid if param_grid is not None else DEFAULT_GRID
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        search = GridSearchCV(
            estimator=self.model,
            param_grid=grid,
            cv=splitter,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        best_index = list(search.cv_results_["params"]).index(search.best_params_)
        fold_scores = []
        for fold in range(cv):
            key = "split%d_test_score" % fold
            if key in search.cv_results_:
                fold_scores.append(float(search.cv_results_[key][best_index]))
        return {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_fold_scores": fold_scores,
            "cv_results_mean": float(
                np.mean(search.cv_results_["mean_test_score"])
            ),
            "cv_results_std": float(np.std(search.cv_results_["mean_test_score"])),
        }

    def save_model(self, model_path: str) -> None:
        """Persist the fitted model and its class mapping.

        Args:
            model_path: Destination path. Parent directories are created.
        """
        parent = os.path.dirname(os.path.abspath(model_path))
        os.makedirs(parent, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "class_to_idx": self.class_to_idx,
                    "idx_to_class": self.idx_to_class,
                },
                f,
            )

    @classmethod
    def load_model(cls, model_path: str) -> "SVMClassifier":
        """Load a model persisted by :meth:`save_model`.

        Args:
            model_path: Path to the pickled model.

        Returns:
            The restored classifier.
        """
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        classifier = cls()
        classifier.model = data["model"]
        classifier.class_to_idx = data["class_to_idx"]
        classifier.idx_to_class = data["idx_to_class"]
        return classifier


def svm_train(
    train_labels: str = "labels/train_labels.json",
    test_labels: str = "labels/test_labels.json",
    model_save_path: str = "results/svm_face_model.pkl",
    image_size: Tuple[int, int] = (32, 32),
    tune: bool = False,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Train and evaluate the SVM, optionally tuning its hyper-parameters.

    The class mapping is derived from the training manifest only and is reused
    for the test manifest, so the two splits always agree on the label
    indices.

    Args:
        train_labels: Path to the training manifest.
        test_labels: Path to the test manifest.
        model_save_path: Where to write the pickled model.
        image_size: Target ``(width, height)`` in pixels.
        tune: Run a cross-validated grid search before the final fit.
        cv: Number of folds used for tuning and cross-validation.
        seed: Seed controlling the fold split.

    Returns:
        A dictionary describing the training run.
    """
    classifier = SVMClassifier()
    X_train, y_train = classifier.load_dataset(train_labels, image_size)
    mapping = classifier.mapping
    X_test, y_test = classifier.load_dataset(test_labels, image_size, mapping=mapping)

    results: Dict[str, object] = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_classes": len(mapping),
    }

    if tune and len(np.unique(y_train)) >= cv:
        tuning = classifier.tune(X_train, y_train, cv=cv, seed=seed)
        results["tuning"] = tuning
        print(f"Best parameters: {tuning['best_params']}")
        print(f"Best CV score: {tuning['best_score']:.4f}")
    else:
        results["tuning"] = None

    classifier.train(X_train, y_train)

    cv_result = classifier.cross_validate(X_train, y_train, cv=cv, seed=seed)
    results["cv_mean"] = cv_result["cv_mean"]
    results["cv_std"] = cv_result["cv_std"]
    results["cv_folds"] = cv_result["cv_folds"]

    train_accuracy = classifier.evaluate(X_train, y_train)
    test_accuracy = classifier.evaluate(X_test, y_test)
    results["train_accuracy"] = train_accuracy
    results["test_accuracy"] = test_accuracy

    classifier.save_model(model_save_path)
    results["model_path"] = model_save_path

    print("Loading dataset...")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        f"Cross-validated accuracy: {cv_result['cv_mean']:.4f} "
        f"+/- {cv_result['cv_std']:.4f}"
    )
    print(f"Model saved to {model_save_path}")
    return results
