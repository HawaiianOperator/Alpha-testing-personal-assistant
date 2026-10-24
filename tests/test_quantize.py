import unittest
import numpy as np
from local_model.network import Network
from local_model.quantize import (
    calibrate, encode_quantized, head_quantized, predict_quantized,
    quantize_input, requantize,
)


class QuantizationTests(unittest.TestCase):
    def test_q24_rounding_and_saturation(self):
        # Half-integer cases verify signed ties-away-from-zero.
        actual = requantize(np.array([-5, -3, -1, 1, 3, 5]), 1 << 23)
        np.testing.assert_array_equal(actual, [-3, -2, -1, 1, 2, 3])
        np.testing.assert_array_equal(
            requantize(np.array([-100000, -1, 100000]), 1 << 24, relu=True),
            [0, 0, 127],
        )

    def test_uint8_input_boundaries(self):
        pixels = np.zeros((1, 1, 64, 64), dtype=np.uint8)
        pixels[0, 0, 0, :4] = [0, 1, 254, 255]
        np.testing.assert_array_equal(quantize_input(pixels)[0, 0, 0, :4], [0, 0, 127, 127])
        with self.assertRaises(ValueError):
            quantize_input(pixels.astype(np.float32))

    def test_partitioned_head_matches_full_integer_inference(self):
        network = Network(seed=123)
        pixels = np.random.default_rng(19).integers(0, 256, (3, 1, 64, 64), dtype=np.uint8)
        params, details = calibrate(network, pixels.astype(np.float32) / 255.0)
        features = encode_quantized(params, pixels)
        self.assertEqual(features.shape, (3, 16))
        self.assertEqual(features.dtype, np.int8)
        np.testing.assert_array_equal(
            predict_quantized(params, pixels), head_quantized(params, features)
        )
        for layer in details["layers"].values():
            self.assertLessEqual(layer["max_abs_accumulator_bound"], np.iinfo(np.int32).max)
        invalid = features.copy()
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            head_quantized(params, invalid)

    def test_bias_overflow_fails_export(self):
        network = Network(seed=2)
        network.params["conv1_b"][:] = 1e20
        with self.assertRaisesRegex(ValueError, "bias exceeds INT32"):
            calibrate(network, np.zeros((1, 1, 64, 64), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
