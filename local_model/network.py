"""A small CNN implemented and trained from scratch using only NumPy.

Inference architecture: 64x64x1 -> valid conv8 -> pool2 -> valid conv16 -> GAP -> dense4.
Arrays use NCHW. No pretrained parameters, remote calls, or ML framework are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


PARAMETER_SHAPES = {
    "conv1_w": (8, 1, 3, 3),
    "conv1_b": (8,),
    "conv2_w": (16, 8, 3, 3),
    "conv2_b": (16,),
    "dense_w": (4, 16),
    "dense_b": (4,),
}
CLASS_NAMES = ("ready", "busy", "done", "other")


def _im2col(x: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
    return np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5)).reshape(-1, x.shape[1] * 9)


def _conv(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    n, _, height, width = x.shape
    oh, ow = height - 2, width - 2
    columns = _im2col(x)
    output = columns @ weight.reshape(weight.shape[0], -1).T
    output += bias
    output = output.reshape(n, oh, ow, weight.shape[0]).transpose(0, 3, 1, 2)
    return output, columns


def _conv_backward(dy, x, weight, columns, compute_input=True):
    n, channels, height, width = x.shape
    oh, ow = height - 2, width - 2
    flattened = np.ascontiguousarray(dy.transpose(0, 2, 3, 1)).reshape(-1, weight.shape[0])
    dw = (flattened.T @ columns).reshape(weight.shape)
    db = flattened.sum(axis=0)
    dx = None
    if compute_input:
        dcols = (flattened @ weight.reshape(weight.shape[0], -1))
        dcols = dcols.reshape(n, oh, ow, channels, 3, 3).transpose(0, 3, 1, 2, 4, 5)
        dx = np.zeros_like(x)
        for ky in range(3):
            for kx in range(3):
                dx[:, :, ky:ky + oh, kx:kx + ow] += dcols[..., ky, kx]
    return dx, dw, db


def _pool(x):
    n, channels, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("Pool input dimensions must be even.")
    oh, ow = height // 2, width // 2
    windows = x.reshape(n, channels, oh, 2, ow, 2).transpose(0, 1, 2, 4, 3, 5)
    windows = windows.reshape(n, channels, oh, ow, 4)
    selected = windows.argmax(axis=-1)
    output = np.take_along_axis(windows, selected[..., None], axis=-1)[..., 0]
    return output, selected


def _pool_backward(dy, selected):
    n, channels, oh, ow = dy.shape
    windows = np.zeros((n, channels, oh, ow, 4), dtype=dy.dtype)
    np.put_along_axis(windows, selected[..., None], dy[..., None], axis=-1)
    return windows.reshape(n, channels, oh, ow, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, channels, oh * 2, ow * 2)


def cross_entropy(logits: np.ndarray, labels: np.ndarray):
    """Return stable mean cross-entropy and its gradient with respect to logits."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(logits) or labels.dtype.kind not in "iu":
        raise ValueError("Labels must be a one-dimensional integer array matching the batch.")
    if np.any(labels < 0) or np.any(labels >= 4):
        raise ValueError("Labels must be in [0, 3].")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = exp_values.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(denominator[:, 0]) - shifted[np.arange(len(labels)), labels])
    gradient = exp_values / denominator
    gradient[np.arange(len(labels)), labels] -= 1
    gradient /= len(labels)
    return float(loss), gradient


class Network:
    def __init__(self, seed: int = 7, params: Mapping[str, np.ndarray] | None = None):
        rng = np.random.default_rng(seed)
        self.params = {}
        for name, shape in PARAMETER_SHAPES.items():
            if name.endswith("_b"):
                value = np.zeros(shape, dtype=np.float32)
            else:
                fan_in = int(np.prod(shape[1:]))
                value = (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
            self.params[name] = value
        if params is not None:
            self.load_state_dict(params)
        self._cache = None

    @property
    def parameter_count(self):
        return sum(value.size for value in self.params.values())

    @staticmethod
    def _validate_input(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (1, 64, 64) or not len(x):
            raise ValueError("Input must have shape (N, 1, 64, 64) with N >= 1.")
        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Inputs must be finite grayscale values in [0, 1].")
        return x

    def _forward(self, x):
        x = self._validate_input(x)
        p = self.params
        pre1, col1 = _conv(x, p["conv1_w"], p["conv1_b"])
        conv1 = np.maximum(pre1, 0)
        pool1, pool_selected = _pool(conv1)
        pre2, col2 = _conv(pool1, p["conv2_w"], p["conv2_b"])
        conv2 = np.maximum(pre2, 0)
        gap = conv2.mean(axis=(2, 3))
        logits = gap @ p["dense_w"].T + p["dense_b"]
        activations = {"input": x, "conv1": conv1, "pool1": pool1,
                       "conv2": conv2, "gap": gap, "logits": logits}
        cache = {"x": x, "pre1": pre1, "col1": col1, "pool1": pool1,
                 "pool_selected": pool_selected, "pre2": pre2, "col2": col2,
                 "conv2": conv2, "gap": gap}
        return logits, activations, cache

    def forward(self, x, cache=False):
        """Return logits. cache=True retains an internal training cache, not a tuple."""
        logits, _, saved = self._forward(x)
        self._cache = saved if cache else None
        return logits

    def forward_activations(self, x):
        """Return post-ReLU activations used for calibration, including final logits."""
        _, activations, _ = self._forward(x)
        return activations

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def loss_and_gradients(self, x, labels):
        logits, _, c = self._forward(x)
        loss, dlogits = cross_entropy(logits, np.asarray(labels))
        p = self.params
        gradients = {
            "dense_w": dlogits.T @ c["gap"],
            "dense_b": dlogits.sum(axis=0),
        }
        dgap = dlogits @ p["dense_w"]
        dconv2 = np.broadcast_to(
            dgap[:, :, None, None] / (c["conv2"].shape[2] * c["conv2"].shape[3]),
            c["conv2"].shape,
        )
        dpre2 = dconv2 * (c["pre2"] > 0)
        dpool1, gradients["conv2_w"], gradients["conv2_b"] = _conv_backward(
            dpre2, c["pool1"], p["conv2_w"], c["col2"]
        )
        dconv1 = _pool_backward(dpool1, c["pool_selected"])
        dpre1 = dconv1 * (c["pre1"] > 0)
        _, gradients["conv1_w"], gradients["conv1_b"] = _conv_backward(
            dpre1, c["x"], p["conv1_w"], c["col1"], compute_input=False
        )
        return loss, gradients

    def state_dict(self):
        return {name: value.copy() for name, value in self.params.items()}

    def load_state_dict(self, params):
        if set(params) != set(PARAMETER_SHAPES):
            raise ValueError("Parameter keys do not match the fixed CNN architecture.")
        for name, shape in PARAMETER_SHAPES.items():
            value = np.asarray(params[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter {name}; expected finite array of shape {shape}.")
            self.params[name] = value.copy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(params={name: data[name] for name in PARAMETER_SHAPES})


class Adam:
    """Adam updates with optional global gradient norm clipping."""
    def __init__(self, params, learning_rate=0.003, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, clip_norm=5.0):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon, self.clip_norm = epsilon, clip_norm
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.step_number = 0

    def step(self, params, grads):
        self.step_number += 1
        norm = np.sqrt(sum(float(np.sum(value.astype(np.float64) ** 2)) for value in grads.values()))
        factor = min(1.0, self.clip_norm / (norm + 1e-12)) if self.clip_norm else 1.0
        for name, param in params.items():
            gradient = grads[name] * factor
            if not np.isfinite(gradient).all():
                raise FloatingPointError("Encountered non-finite training gradient.")
            self.m[name] *= self.beta1
            self.m[name] += (1 - self.beta1) * gradient
            self.v[name] *= self.beta2
            self.v[name] += (1 - self.beta2) * gradient * gradient
            corrected_m = self.m[name] / (1 - self.beta1 ** self.step_number)
            corrected_v = self.v[name] / (1 - self.beta2 ** self.step_number)
            param -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + self.epsilon)
