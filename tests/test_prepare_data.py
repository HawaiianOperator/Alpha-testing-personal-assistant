import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from local_model.capture import _capture_directory, validate_bbox, validate_session_name
from local_model.dataset import CLASS_NAMES
from local_model.prepare_data import prepare_dataset


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for session, value in (("train_a", 40), ("val_a", 100), ("test_a", 180)):
            for index, label in enumerate(CLASS_NAMES):
                path = self.root / session / label / "one.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (80 + index, 72), (value, value, value)).save(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_explicit_splits_shapes_values_and_metadata(self):
        data = prepare_dataset(self.root, ["train_a"], ["val_a"], ["test_a"])
        for split in ("train", "val", "test"):
            self.assertEqual(data[f"x_{split}"].shape, (4, 1, 64, 64))
            self.assertEqual(data[f"x_{split}"].dtype, np.float32)
            self.assertEqual(data[f"y_{split}"].tolist(), [0, 1, 2, 3])
            self.assertEqual(data[f"y_{split}"].dtype, np.int64)
            self.assertGreaterEqual(float(data[f"x_{split}"].min()), 0)
            self.assertLessEqual(float(data[f"x_{split}"].max()), 1)
            self.assertEqual(set(data[f"session_{split}"]), {f"{split}_a"})
        self.assertAlmostEqual(float(data["x_train"][0, 0, 0, 0]), 40 / 255, places=6)
        self.assertAlmostEqual(float(data["x_test"][0, 0, 0, 0]), 180 / 255, places=6)
        metadata = json.loads(str(data["metadata_json"]))
        self.assertFalse(metadata["synthetic"])
        self.assertTrue(metadata["real_desktop_data"])
        self.assertEqual(metadata["split_sessions"]["test"], ["test_a"])
        self.assertEqual(metadata["samples"], 12)

    def test_overlap_or_repeat_is_rejected(self):
        for train, val, test in (
            (["train_a"], ["train_a"], ["test_a"]),
            (["train_a", "train_a"], ["val_a"], ["test_a"]),
        ):
            with self.subTest(train=train, val=val), self.assertRaisesRegex(ValueError, "disjoint"):
                prepare_dataset(self.root, train, val, test)

    def test_missing_training_class_is_rejected(self):
        (self.root / "train_a" / "OTHER" / "one.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing classes: OTHER"):
            prepare_dataset(self.root, ["train_a"], ["val_a"], ["test_a"])

    def test_empty_evaluation_or_missing_session_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "val has no PNG"):
            prepare_dataset(self.root, ["train_a"], ["empty"], ["test_a"])
        with self.assertRaisesRegex(ValueError, "does not exist"):
            prepare_dataset(self.root, ["train_a"], ["missing"], ["test_a"])

    def test_session_traversal_and_invalid_names_are_rejected(self):
        for name in ("../outside", "/absolute", "a/b", "a\\b", ".", "", "bad name", "x" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare_dataset(self.root, [name], ["val_a"], ["test_a"])
            with self.subTest(capture=name), self.assertRaises(ValueError):
                validate_session_name(name)

    def test_corrupt_png_and_sample_cap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample preparation limit"):
            prepare_dataset(self.root, ["train_a"], ["val_a"], ["test_a"], max_samples=11)
        (self.root / "train_a" / "READY" / "one.png").write_bytes(b"not a png")
        with self.assertRaisesRegex(ValueError, "Could not read PNG"):
            prepare_dataset(self.root, ["train_a"], ["val_a"], ["test_a"])

    def test_capture_region_validation_does_not_capture_screen(self):
        self.assertEqual(validate_bbox((-100, 0, 100, 200)), (-100, 0, 100, 200))
        for bbox in ((0, 0, 0, 1), (1, 1, 0, 3), (0, 0, 10000, 10000), (0, 0, 3.5, 5)):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                validate_bbox(bbox)
        path = _capture_directory(self.root, "capture_01", "READY")
        self.assertEqual(path, self.root / "capture_01" / "READY")
        with self.assertRaises(ValueError):
            _capture_directory(self.root, "../escape", "READY")

    def test_empty_lists_are_not_implicitly_split(self):
        with self.assertRaisesRegex(ValueError, "val must explicitly"):
            prepare_dataset(self.root, ["train_a"], [], ["test_a"])


if __name__ == "__main__":
    unittest.main()
