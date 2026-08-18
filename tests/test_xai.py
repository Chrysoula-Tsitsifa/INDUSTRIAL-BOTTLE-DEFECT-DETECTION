from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from app.core.validation import InputValidationError
from app.core.xai import (
    generate_gradcam,
    load_gradcam_models,
)


class XAITests(unittest.TestCase):

    def test_gradcam_models_match_expected_architecture(self) -> None:
        models = load_gradcam_models()

        self.assertEqual(
            models.target_layer,
            "out_relu",
        )

        self.assertEqual(
            models.target_signal,
            "SUCCESS",
        )

        self.assertEqual(
            models.conv_model.output_shape,
            (None, 7, 7, 1280),
        )

        self.assertEqual(
            models.classifier_model.input_shape,
            (None, 7, 7, 1280),
        )

        self.assertEqual(
            models.classifier_model.output_shape,
            (None, 1),
        )

    def test_generate_gradcam_returns_valid_evidence(self) -> None:
        image = Image.new(
            "RGB",
            (640, 480),
            (128, 64, 32),
        )

        result = generate_gradcam(
            image
        )

        self.assertEqual(
            result.heatmap.shape,
            (7, 7),
        )

        self.assertEqual(
            result.heatmap.dtype,
            np.float32,
        )

        self.assertTrue(
            np.isfinite(
                result.heatmap
            ).all()
        )

        self.assertGreaterEqual(
            float(result.heatmap.min()),
            0.0,
        )

        self.assertLessEqual(
            float(result.heatmap.max()),
            1.0,
        )

        self.assertAlmostEqual(
            float(result.heatmap.max()),
            1.0,
            places=5,
        )

        self.assertEqual(
            result.overlay.size,
            image.size,
        )

        self.assertEqual(
            result.overlay.mode,
            "RGB",
        )

        self.assertIn(
            result.target_class,
            (0, 1),
        )

        expected_label = (
            "Defective"
            if result.target_class == 0
            else "Good"
        )

        self.assertEqual(
            result.target_label,
            expected_label,
        )

        self.assertEqual(
            result.target_layer,
            "out_relu",
        )

        self.assertEqual(
            result.target_signal,
            "SUCCESS",
        )

    def test_gradcam_rejects_too_small_image(self) -> None:
        image = Image.new(
            "RGB",
            (16, 16),
        )

        with self.assertRaises(
            InputValidationError
        ):
            generate_gradcam(
                image
            )


if __name__ == "__main__":
    unittest.main()