from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from app.core.preprocessing import (
    ImagePreprocessingError,
    decode_image,
    prepare_mobilenet_input,
    prepare_reconstruction_input,
    prepare_svm_input,
)


class PreprocessingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.image = Image.new(
            "RGB",
            (640, 480),
            (128, 64, 32),
        )

    def test_reconstruction_input_contract(self) -> None:
        batch = prepare_reconstruction_input(self.image)

        self.assertEqual(batch.shape, (1, 128, 128, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertGreaterEqual(float(batch.min()), 0.0)
        self.assertLessEqual(float(batch.max()), 1.0)

    def test_svm_input_contract(self) -> None:
        features = prepare_svm_input(self.image)

        self.assertEqual(features.shape, (1, 49152))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_mobilenet_input_contract(self) -> None:
        batch = prepare_mobilenet_input(self.image)

        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(np.isfinite(batch).all())

        self.assertGreaterEqual(float(batch.min()), -1.0)
        self.assertLessEqual(float(batch.max()), 1.0)

    def test_decode_valid_image(self) -> None:
        buffer = BytesIO()
        self.image.save(buffer, format="PNG")

        decoded = decode_image(buffer.getvalue())

        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (640, 480))

    def test_decode_rejects_empty_payload(self) -> None:
        with self.assertRaises(ImagePreprocessingError):
            decode_image(b"")

    def test_decode_rejects_invalid_payload(self) -> None:
        with self.assertRaises(ImagePreprocessingError):
            decode_image(b"this-is-not-an-image")


if __name__ == "__main__":
    unittest.main()