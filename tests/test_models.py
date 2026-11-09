"""Tests for the neural network and the SVM classifier."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import build_class_mapping
from src.training.svm_classifier import SVMClassifier
from src.utils.utils import BPNeuralNetwork, preprocess_image, softmax


def test_sigmoid_does_not_overflow():
    """Large negative inputs must not produce nan or a warning."""
    net = BPNeuralNetwork(2, 2, 2)
    with np.errstate(over="raise", invalid="raise"):
        values = net.sigmoid(np.array([-1000.0, -800.0, 0.0, 800.0]))
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    # The saturated ends round to 0 and 1, the centre to one half.
    assert values[0] == 0.0
    assert values[3] == 1.0
    assert values[2] == 0.5


def test_sigmoid_is_monotone_in_the_representable_range():
    """Within the representable range the sigmoid must stay strictly monotone."""
    net = BPNeuralNetwork(2, 2, 2)
    values = net.sigmoid(np.array([-10.0, -1.0, 0.0, 1.0, 10.0]))
    assert np.all(np.diff(values) > 0)


def test_sigmoid_derivative_is_consistent():
    """The derivative must match a finite difference of the sigmoid."""
    net = BPNeuralNetwork(2, 2, 2)
    x = np.array([[-1.0, 0.0, 1.0]])
    analytic = net.sigmoid_deriv(net.sigmoid(x))
    eps = 1e-6
    numeric = (net.sigmoid(x + eps) - net.sigmoid(x - eps)) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_softmax_rows_sum_to_one():
    """Each softmax row must be a probability distribution."""
    logits = np.array([[1.0, 2.0, 3.0], [-10.0, 0.0, 10.0]])
    result = softmax(logits)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.all(np.isfinite(result))


def test_softmax_is_shift_invariant():
    """Adding a constant to the logits must not change the result."""
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(logits), softmax(logits + 7.5))


def test_network_training_reduces_loss():
    """Training on an easy problem must reduce the loss."""
    rng = np.random.default_rng(0)
    X = rng.random((64, 8))
    y = np.zeros((64, 2))
    y[:, 0] = (X[:, 0] > 0.5).astype(float)
    y[:, 1] = 1 - y[:, 0]

    net = BPNeuralNetwork(8, 16, 2)
    history = net.train(X, y, epochs=60, learning_rate=0.5, batch_size=16, seed=1)
    assert history[-1][1] < history[0][1]


def test_network_training_is_reproducible():
    """The same seed must produce the same loss history."""
    rng = np.random.default_rng(0)
    X = rng.random((32, 4))
    y = np.eye(2)[np.arange(32) % 2]

    first = BPNeuralNetwork(4, 8, 2, seed=3)
    first.train(X, y, epochs=20, learning_rate=0.1, batch_size=8, seed=3)
    second = BPNeuralNetwork(4, 8, 2, seed=3)
    second.train(X, y, epochs=20, learning_rate=0.1, batch_size=8, seed=3)
    assert first.loss_history == second.loss_history


def test_preprocess_image_shape_and_range(manifests):
    """The preprocessed vector must have the expected length and range."""
    import json

    with open(manifests["train"], "r", encoding="utf-8") as f:
        entries = json.load(f)
    vector = preprocess_image(entries[0]["path"], (32, 32))
    assert vector.shape == (32 * 32 * 3,)
    assert vector.min() >= 0.0 and vector.max() <= 1.0


def test_svm_shares_mapping_between_splits(manifests):
    """Regression test: both splits must agree on the label indices."""
    classifier = SVMClassifier()
    X_train, y_train = classifier.load_dataset(manifests["train"], (16, 16))
    mapping = classifier.mapping
    X_test, y_test = classifier.load_dataset(
        manifests["test"], (16, 16), mapping=mapping
    )

    assert classifier.class_to_idx == mapping.class_to_idx
    # Every test label must be decodable back to a label the training set saw.
    for index in np.unique(y_test):
        assert classifier.idx_to_class[int(index)] in classifier.class_to_idx
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_svm_trains_and_predicts(manifests):
    """The SVM must fit and produce predictions for the test split."""
    classifier = SVMClassifier(C=10.0, gamma="scale")
    X_train, y_train = classifier.load_dataset(manifests["train"], (16, 16))
    mapping = classifier.mapping
    X_test, y_test = classifier.load_dataset(
        manifests["test"], (16, 16), mapping=mapping
    )
    classifier.train(X_train, y_train)
    predictions = classifier.predict(X_test)
    assert len(predictions) == len(y_test)
    assert 0.0 <= classifier.evaluate(X_test, y_test) <= 1.0


def test_svm_cross_validate_returns_mean_and_std(manifests):
    """Cross-validation must report a mean and a standard deviation."""
    classifier = SVMClassifier()
    X_train, y_train = classifier.load_dataset(manifests["train"], (16, 16))
    result = classifier.cross_validate(X_train, y_train, cv=3, seed=1)
    assert "cv_mean" in result and "cv_std" in result
    assert len(result["cv_folds"]) == 3


def test_svm_model_roundtrip(manifests, tmp_path):
    """A saved model must reload with the same class mapping."""
    classifier = SVMClassifier()
    X_train, y_train = classifier.load_dataset(manifests["train"], (16, 16))
    classifier.train(X_train, y_train)

    path = tmp_path / "model.pkl"
    classifier.save_model(str(path))
    restored = SVMClassifier.load_model(str(path))
    assert restored.class_to_idx == classifier.class_to_idx
    assert restored.idx_to_class == classifier.idx_to_class


def test_build_class_mapping_is_stable():
    """Repeated calls with the same labels must give the same mapping."""
    first = build_class_mapping(["b", "a"])
    second = build_class_mapping(["a", "b"])
    assert first.class_to_idx == second.class_to_idx
