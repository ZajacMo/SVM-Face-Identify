"""Shared utilities: image preprocessing and the from-scratch BP neural network.

This module deliberately implements the multi-layer perceptron with NumPy only,
so the repository has no heavy deep-learning dependency and the training loop is
transparent to readers.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

__all__ = ["BPNeuralNetwork", "preprocess_image", "iter_batches", "softmax"]


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable row-wise softmax.

    Args:
        x: 2-D array of shape ``(n_samples, n_classes)`` of logits.

    Returns:
        Array of the same shape whose rows sum to one.
    """
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def preprocess_image(image_path: str, image_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load an image and flatten it into a normalised feature vector.

    Args:
        image_path: Path to the image file. Converted to RGB, so grayscale and
            palette images are handled identically to colour ones.
        image_size: Target ``(width, height)`` in pixels.

    Returns:
        1-D float array of length ``width * height * 3`` with values in
        ``[0.0, 1.0]``.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(image_size)
        array = np.asarray(img, dtype=np.float64) / 255.0
    return array.flatten()


def iter_batches(
    size: int, batch_size: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield shuffled index batches for mini-batch training.

    Args:
        size: Total number of samples.
        batch_size: Maximum number of samples per batch.

    Yields:
        A pair of index arrays ``(indices, complement)``; the second array holds
        the indices not drawn in this batch. The permutation is regenerated for
        every epoch, so the caller controls reproducibility through the random
        state.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    permutation = np.random.permutation(size)
    for start in range(0, size, batch_size):
        batch = permutation[start : start + batch_size]
        complement = permutation[start + batch_size :]
        yield batch, complement


class BPNeuralNetwork:
    """A fully connected multi-layer perceptron trained with backpropagation.

    The network has a single hidden layer with a sigmoid activation and a
    softmax output layer, trained with cross-entropy loss. The output layer
    uses the standard identity-plus-softmax formulation, for which the
    derivative of the loss with respect to the logits is simply
    ``y - output``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise weights and biases.

        Args:
            input_size: Number of input features.
            hidden_size: Number of units in the hidden layer.
            output_size: Number of output classes.
            seed: Optional seed for the weight initialisation, so that two
                networks built with the same seed start from the same state.

        Raises:
            ValueError: If any dimension is not positive.
        """
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("layer sizes must be positive")
        if seed is not None:
            np.random.seed(seed)
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
        self.hidden: np.ndarray | None = None
        self.output: np.ndarray | None = None
        self.loss_history: list[tuple[int, float]] = []
        self.learning_rate: float = 0.0

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply the logistic sigmoid safely.

        The exponential is evaluated on the non-positive branch only, so no
        input can trigger an overflow and produce ``nan`` gradients. Inputs
        that would overflow are mathematically indistinguishable from zero to
        double precision, so the result stays exact and strictly monotone.

        Args:
            x: Input array of any shape.

        Returns:
            Array with the same shape as ``x`` containing values in ``[0, 1]``.
        """
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        positive = x >= 0
        exp_neg = np.exp(-x[positive])
        out[positive] = 1.0 / (1.0 + exp_neg)
        exp_pos = np.exp(x[~positive])
        out[~positive] = exp_pos / (1.0 + exp_pos)
        return out

    @staticmethod
    def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
        """Return the derivative of the sigmoid given its output.

        Args:
            x: Activations produced by :meth:`sigmoid`.

        Returns:
            Element-wise derivative values.
        """
        return x * (1.0 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run the forward pass.

        Args:
            X: Input matrix of shape ``(n_samples, input_size)``.

        Returns:
            The softmax output of shape ``(n_samples, output_size)``.
        """
        X = np.asarray(X, dtype=np.float64)
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        logits = np.dot(self.hidden, self.weights2) + self.bias2
        self.output = softmax(logits)
        return self.output

    def backward(
        self, X: np.ndarray, y: np.ndarray, output: np.ndarray, learning_rate: float
    ) -> None:
        """Update weights and biases from a batch of gradients.

        Args:
            X: Batch of inputs, shape ``(batch, input_size)``.
            y: Batch of one-hot targets, shape ``(batch, output_size)``.
            output: Forward-pass output for ``X``.
            learning_rate: Step size for the gradient update.
        """
        output_error = y - output
        output_delta = output_error

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_deriv(self.hidden)

        self.weights2 += self.hidden.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        seed: int | None = None,
    ) -> list[tuple[int, float]]:
        """Train the network with mini-batch gradient descent.

        Args:
            X: Training inputs of shape ``(n_samples, input_size)``.
            y: One-hot encoded targets of shape ``(n_samples, output_size)``.
            epochs: Number of passes over the data.
            learning_rate: Step size for the gradient update.
            batch_size: Number of samples per gradient step.
            seed: Optional seed applied before shuffling, so a run is
                reproducible.

        Returns:
            The loss history as ``(epoch, loss)`` pairs, sampled every 100
            epochs and always including the final epoch.
        """
        if seed is not None:
            np.random.seed(seed)
        self.loss_history = []
        self.learning_rate = learning_rate
        n_samples = X.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start : start + batch_size]
                y_batch = y_shuffled[start : start + batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)
            if epoch % 100 == 0 or epoch == epochs - 1:
                output = self.forward(X)
                loss = -np.mean(y * np.log(output + 1e-10))
                self.loss_history.append((epoch, float(loss)))
        return self.loss_history
