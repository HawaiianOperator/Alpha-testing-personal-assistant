"""Numerical checks for the custom CNN, using no test-only dependencies."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from local_model.network import Adam, Network, cross_entropy


class NetworkTests(unittest.TestCase):
    def test_shapes_parameters_and_round_trip(self):
        network = Network(seed=11)
        x = np.random.default_rng(1).random((2, 1, 64, 64), dtype=np.float32)
        activations = network.forward_activations(x)
        self.assertEqual(network.parameter_count, 1316)
        self.assertEqual(activations["conv1"].shape, (2, 8, 62, 62))
        self.assertEqual(activations["pool1"].shape, (2, 8, 31, 31))
        self.assertEqual(activations["conv2"].shape, (2, 16, 29, 29))
        self.assertEqual(activations["logits"].shape, (2, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.npz"
            network.save(path)
            restored = Network.load(path)
            np.testing.assert_array_equal(network.forward(x), restored.forward(x))

    def test_analytical_gradients_match_finite_differences(self):
        network = Network(seed=19)
        # Double-precision parameters make numerical gradient comparisons precise.
        network.params = {name: value.astype(np.float64) for name, value in network.params.items()}
        x = np.random.default_rng(37).uniform(0.1, 0.9, (1, 1, 64, 64)).astype(np.float32)
        y = np.array([2], dtype=np.int64)
        _, analytical = network.loss_and_gradients(x, y)
        selections = {
            "conv1_w": [(0, 0, 0, 0), (6, 0, 1, 2)],
            "conv1_b": [(2,)],
            "conv2_w": [(0, 0, 0, 0), (12, 5, 1, 1)],
            "conv2_b": [(5,)],
            "dense_w": [(2, 5), (0, 1)],
            "dense_b": [(1,)],
        }
        epsilon = 1e-5
        for name, indices in selections.items():
            for index in indices:
                original = network.params[name][index]
                network.params[name][index] = original + epsilon
                high, _ = cross_entropy(network.forward(x), y)
                network.params[name][index] = original - epsilon
                low, _ = cross_entropy(network.forward(x), y)
                network.params[name][index] = original
                numerical = (high - low) / (2 * epsilon)
                self.assertAlmostEqual(float(analytical[name][index]), numerical, delta=2e-5,
                                       msg=f"Gradient mismatch for {name}{index}")

    def test_update_reduces_loss_on_same_batch(self):
        network = Network(seed=4)
        x = np.random.default_rng(3).random((2, 1, 64, 64), dtype=np.float32)
        y = np.array([0, 0], dtype=np.int64)
        before, gradients = network.loss_and_gradients(x, y)
        Adam(network.params, learning_rate=0.0001).step(network.params, gradients)
        after, _ = cross_entropy(network.forward(x), y)
        self.assertLess(after, before)

    def test_rejects_bad_inputs_and_labels(self):
        network = Network()
        with self.assertRaises(ValueError):
            network.forward(np.zeros((1, 64, 64), dtype=np.float32))
        with self.assertRaises(ValueError):
            network.forward(np.full((1, 1, 64, 64), np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            network.loss_and_gradients(np.zeros((1, 1, 64, 64), dtype=np.float32),
                                       np.array([4], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
