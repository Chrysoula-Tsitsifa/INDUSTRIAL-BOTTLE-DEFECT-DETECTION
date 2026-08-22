from __future__ import annotations

import unittest

from PIL import Image

from app.core.validation import (
    ImageValidationResult,
    InputValidationError,
    validate_image_for_inference,
)


class ValidationTests(unittest.TestCase):

    def test_valid_rgb_image_passes(self) -> None:
        image = Image.new(
            "RGB",
            (640, 480),
            (120, 80, 40),
        )

        result = validate_image_for_inference(image)

        self.assertIsInstance(
            result,
            ImageValidationResult,
        )

        self.assertEqual(
            result.width,
            640,
        )

        self.assertEqual(
            result.height,
            480,
        )

        self.assertEqual(
            result.channels,
            3,
        )

        self.assertEqual(
            result.mode,
            "RGB",
        )

    def test_grayscale_image_is_accepted_as_rgb_compatible(self) -> None:
        image = Image.new(
            "L",
            (128, 128),
            120,
        )

        result = validate_image_for_inference(image)

        self.assertEqual(
            result.channels,
            3,
        )

        self.assertEqual(
            result.mode,
            "RGB",
        )

    def test_too_small_image_is_rejected(self) -> None:
        image = Image.new(
            "RGB",
            (16, 16),
        )

        with self.assertRaises(InputValidationError):
            validate_image_for_inference(image)

    def test_invalid_object_is_rejected(self) -> None:
        with self.assertRaises(InputValidationError):
            validate_image_for_inference(
                "not-an-image"
            )


if __name__ == "__main__":
    unittest.main()