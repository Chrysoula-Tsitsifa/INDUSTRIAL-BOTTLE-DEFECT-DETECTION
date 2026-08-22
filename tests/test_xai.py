from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from app.core.preprocessing import prepare_mobilenet_input
from app.core.validation import InputValidationError
from app.core.xai import (
    XAI_CANDIDATE_LAYERS,
    XAIError,
    _make_gradcam_pp_heatmap,
    generate_gradcam,
    load_gradcam_models,
)


class XAITests(unittest.TestCase):

    def test_candidate_layers_have_expected_spatial_resolution(self) -> None:
        expected_shapes = {
            "block_5_add": (None, 28, 28, 32),
            "block_12_add": (None, 14, 14, 96),
            "out_relu": (None, 7, 7, 1280),
        }

        self.assertEqual(
            XAI_CANDIDATE_LAYERS,
            (
                "block_5_add",
                "block_12_add",
                "out_relu",
            ),
        )

        for layer_name, expected_shape in expected_shapes.items():
            with self.subTest(
                layer=layer_name
            ):
                models = load_gradcam_models(
                    layer_name
                )

                self.assertEqual(
                    models.target_layer,
                    layer_name,
                )

                self.assertEqual(
                    models.target_signal,
                    "SUCCESS",
                )

                self.assertEqual(
                    models.conv_model.output_shape,
                    expected_shape,
                )

                self.assertEqual(
                    models.gradient_model.output_shape[0],
                    expected_shape,
                )

                self.assertEqual(
                    models.gradient_model.output_shape[1],
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

    def test_default_gradcam_layer_remains_out_relu(self) -> None:
        models = load_gradcam_models()

        self.assertEqual(
            models.target_layer,
            "out_relu",
        )

        self.assertEqual(
            models.conv_model.output_shape,
            (None, 7, 7, 1280),
        )

    def test_invalid_gradcam_layer_fails_fast(self) -> None:
        with self.assertRaises(
            XAIError
        ):
            load_gradcam_models(
                "definitely_not_a_real_layer"
            )

    def test_generate_gradcam_returns_valid_default_evidence(self) -> None:
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

    def test_gradcam_pp_returns_valid_out_relu_heatmap(self) -> None:
        image = Image.new(
            "RGB",
            (640, 480),
            (128, 64, 32),
        )

        image_batch = prepare_mobilenet_input(
            image
        )

        models = load_gradcam_models(
            "out_relu"
        )

        heatmap = _make_gradcam_pp_heatmap(
            image_batch,
            models,
            target_class=0,
        )

        self.assertEqual(
            heatmap.shape,
            (7, 7),
        )

        self.assertEqual(
            heatmap.dtype,
            np.float32,
        )

        self.assertTrue(
            np.isfinite(
                heatmap
            ).all()
        )

        self.assertGreaterEqual(
            float(heatmap.min()),
            0.0,
        )

        self.assertLessEqual(
            float(heatmap.max()),
            1.0,
        )

        self.assertAlmostEqual(
            float(heatmap.max()),
            1.0,
            places=5,
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