"""Synthetic unit and integration checks; these are not real-road validation."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import lane_detection as ld

TEST_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def synthetic_lanes(width=320, height=240, missing="none"):
    image = np.full((height, width, 3), 65, dtype=np.uint8)
    if missing != "left":
        cv2.line(image, (int(width*.25), height-1), (int(width*.43), int(height*.42)), (255,255,255), 5)
    if missing != "right":
        cv2.line(image, (int(width*.75), height-1), (int(width*.57), int(height*.42)), (255,255,255), 5)
    return image


class LanePipelineTests(unittest.TestCase):
    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError): ld.load_image("does-not-exist.png")

    def test_loading_and_small_dimensions(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT_ROOT) as folder:
            path = Path(folder) / "small.png"
            cv2.imwrite(str(path), synthetic_lanes(100, 80))
            loaded = ld.load_image(path)
            self.assertEqual(loaded.shape, (80, 100, 3))
            self.assertEqual(ld.process_image(loaded).final_mask.shape, (80, 100))

    def test_binary_polarity(self):
        result = ld.process_image(synthetic_lanes())
        self.assertIsNone(result.fallback)
        self.assertEqual(set(np.unique(result.final_mask)), {0, 255})
        self.assertEqual(int(result.final_mask[-1, 160]), 0)
        self.assertEqual(int(result.final_mask[-1, 20]), 255)

    def test_missing_boundary_uses_fallback(self):
        result = ld.process_image(synthetic_lanes(missing="right"))
        self.assertIsNotNone(result.fallback)
        self.assertEqual(set(np.unique(result.final_mask)), {0, 255})

    def test_no_boundaries_fails(self):
        with self.assertRaisesRegex(RuntimeError, "No lane boundaries") as caught:
            ld.process_image(np.zeros((100, 160, 3), dtype=np.uint8))
        self.assertIn("roi_edges", caught.exception.diagnostics)
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT_ROOT) as folder:
            ld.save_failure_diagnostics(caught.exception.diagnostics, folder)
            self.assertTrue((Path(folder) / "09_detection_failure.png").is_file())
            self.assertFalse((Path(folder) / "08_final_binary_mask.png").exists())

    def test_official_image_keeps_lane_right_of_center_marking(self):
        image = ld.load_image(Path(__file__).resolve().parents[1] / "input" / "official_ps1_road.jpg")
        result = ld.process_image(image)
        self.assertIsNotNone(result.fallback)
        height, width = result.final_mask.shape
        self.assertEqual(int(result.final_mask[height - 10, int(width * .85)]), 0)
        self.assertEqual(int(result.final_mask[height - 10, int(width * .40)]), 255)

    def test_output_written(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT_ROOT) as folder:
            result = ld.process_image(synthetic_lanes())
            path = Path(folder) / "mask.png"
            ld.validate_and_save_mask(result.final_mask, path, (240, 320))
            reopened = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(reopened.shape, (240, 320))
            self.assertEqual(set(np.unique(reopened)), {0, 255})


if __name__ == "__main__": unittest.main()
