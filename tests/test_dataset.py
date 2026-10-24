import json
import unittest

import numpy as np

from local_model.dataset import CLASS_NAMES, build_dataset, make_contact_sheet, render_state


class DatasetTests(unittest.TestCase):
    def test_render_is_deterministic_and_state_changes_only_geometry(self):
        images = [np.asarray(render_state(name, 123, session_seed=789)) for name in CLASS_NAMES]
        np.testing.assert_array_equal(images[0], np.asarray(render_state("READY", 123, session_seed=789)))
        for image in images:
            self.assertEqual(image.shape, (64, 64))
            self.assertEqual(image.dtype, np.uint8)
        # Palette, window chrome and outside noise do not encode the class.
        for image in images[1:]:
            np.testing.assert_array_equal(images[0][:15], image[:15])
        self.assertEqual(len({image.tobytes() for image in images}), len(CLASS_NAMES))

    def test_balanced_splits_and_disjoint_sessions(self):
        data = build_dataset(12, seed=9)
        expected = {"train": 48, "val": 16, "test": 16}
        observed_sessions = []
        image_sets = []
        for split, count in expected.items():
            x, y = data[f"x_{split}"], data[f"y_{split}"]
            self.assertEqual(x.shape, (count, 1, 64, 64))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.int64)
            self.assertGreaterEqual(float(x.min()), 0.0)
            self.assertLessEqual(float(x.max()), 1.0)
            self.assertEqual(np.bincount(y, minlength=4).tolist(), [count // 4] * 4)
            observed_sessions.append(set(data[f"session_{split}"].tolist()))
            image_sets.append({sample.tobytes() for sample in x})
        for first in range(3):
            for second in range(first + 1, 3):
                self.assertFalse(observed_sessions[first] & observed_sessions[second])
                self.assertFalse(image_sets[first] & image_sets[second])
        metadata = json.loads(str(data["metadata_json"]))
        self.assertTrue(metadata["synthetic"])
        self.assertFalse(metadata["real_desktop_data"])
        self.assertEqual(metadata["class_names"], list(CLASS_NAMES))

    def test_dataset_reproducibility(self):
        first = build_dataset(4, 21)
        second = build_dataset(4, 21)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_invalid_state_or_dimensions_are_rejected(self):
        for state in ("missing", -1, 4, True):
            with self.subTest(state=state), self.assertRaises(ValueError):
                render_state(state, 1)
        for size in (0, 15, 1025, 64.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                render_state("READY", 1, size=size)
        with self.assertRaises(ValueError):
            build_dataset(0)

    def test_contact_sheet_contains_four_labeled_columns(self):
        preview = make_contact_sheet(7, samples_per_class=3)
        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, (384, 330))


if __name__ == "__main__":
    unittest.main()
