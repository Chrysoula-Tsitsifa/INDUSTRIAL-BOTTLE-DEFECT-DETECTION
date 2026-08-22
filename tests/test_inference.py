from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from app.core.inference import (
    MODEL_OPTIONS,
    InferenceError,
    load_runtime_models,
    run_inference,
)


class InferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.image = Image.new(
            "RGB",
            (640, 480),
            (128, 64, 32),
        )

        cls.runtime = load_runtime_models()

    def test_runtime_models_load_with_expected_io_contracts(self) -> None:
        self.assertEqual(
            self.runtime.simple_ae.input_shape,
            (None, 128, 128, 3),
        )
        self.assertEqual(
            self.runtime.simple_ae.output_shape,
            (None, 128, 128, 3),
        )

        self.assertEqual(
            self.runtime.tuned_ae.input_shape,
            (None, 128, 128, 3),
        )
        self.assertEqual(
            self.runtime.tuned_ae.output_shape,
            (None, 128, 128, 3),
        )

        self.assertEqual(
            self.runtime.mobilenet.input_shape,
            (None, 224, 224, 3),
        )
        self.assertEqual(
            self.runtime.mobilenet.output_shape,
            (None, 1),
        )

        self.assertEqual(
            sorted(self.runtime.contracts.keys()),
            [
                "notebook1",
                "notebook2",
                "notebook3",
                "notebook4",
            ],
        )

    def test_all_inference_engines_return_consistent_decisions(self) -> None:
        for model_key in MODEL_OPTIONS:

            with self.subTest(model_key=model_key):

                result = run_inference(
                    self.image,
                    model_key,
                )

                self.assertEqual(
                    result.model_key,
                    model_key,
                )

                self.assertIn(
                    result.label,
                    {"Good", "Defective"},
                )

                self.assertTrue(
                    np.isfinite(result.score)
                )

                self.assertTrue(
                    np.isfinite(result.threshold)
                )

                if model_key == "mobilenet_v2":
                    expected_defective = (
                        result.score < result.threshold
                    )
                else:
                    expected_defective = (
                        result.score > result.threshold
                    )

                self.assertEqual(
                    result.is_defective,
                    expected_defective,
                )

                self.assertEqual(
                    result.label,
                    "Defective"
                    if expected_defective
                    else "Good",
                )

    def test_mobilenet_returns_calibrated_probability_metadata(self) -> None:
        result = run_inference(
            self.image,
            "mobilenet_v2",
        )

        self.assertIsNotNone(
            result.raw_probability_good
        )
        self.assertIsNotNone(
            result.calibrated_probability_good
        )

        self.assertGreaterEqual(
            result.raw_probability_good,
            0.0,
        )
        self.assertLessEqual(
            result.raw_probability_good,
            1.0,
        )

        self.assertGreaterEqual(
            result.calibrated_probability_good,
            0.0,
        )
        self.assertLessEqual(
            result.calibrated_probability_good,
            1.0,
        )

        self.assertAlmostEqual(
            result.score,
            result.calibrated_probability_good,
        )

    def test_ssim_returns_structural_similarity_metadata(self) -> None:
        result = run_inference(
            self.image,
            "ssim",
        )

        self.assertIsNotNone(
            result.ssim_similarity
        )

        self.assertTrue(
            np.isfinite(result.ssim_similarity)
        )

        self.assertAlmostEqual(
            result.score,
            1.0 - result.ssim_similarity,
        )

    def test_unknown_model_key_is_rejected(self) -> None:
        with self.assertRaises(InferenceError):
            run_inference(
                self.image,
                "not-a-real-model",
            )


if __name__ == "__main__":
    unittest.main()