from __future__ import annotations

from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class ImagePreprocessingError(ValueError):
    """Raised when an uploaded image cannot satisfy the deployment input contract."""


RECONSTRUCTION_SIZE: Final[tuple[int, int]] = (128, 128)
MOBILENET_SIZE: Final[tuple[int, int]] = (224, 224)

RGB_CHANNELS: Final[int] = 3


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decode uploaded bytes into a validated RGB PIL image.

    The function performs image decoding only. Model-specific resizing and
    numerical transformations are handled by dedicated preprocessing functions.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ImagePreprocessingError(
            "Image payload must be bytes or bytearray."
        )

    if not image_bytes:
        raise ImagePreprocessingError("Image payload is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()

            if image.width <= 0 or image.height <= 0:
                raise ImagePreprocessingError(
                    "Decoded image has invalid dimensions."
                )

            rgb_image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagePreprocessingError(
            "Uploaded file is not a valid decodable image."
        ) from exc

    return rgb_image


def _resize_rgb(
    image: Image.Image,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize an RGB PIL image and return a float32 HWC tensor."""
    if not isinstance(image, Image.Image):
        raise ImagePreprocessingError(
            "Expected a PIL.Image.Image instance."
        )

    resized = image.convert("RGB").resize(
        target_size,
        resample=Image.Resampling.NEAREST,
    )

    array = np.asarray(resized, dtype=np.float32)

    expected_shape = (
        target_size[1],
        target_size[0],
        RGB_CHANNELS,
    )

    if array.shape != expected_shape:
        raise ImagePreprocessingError(
            f"Unexpected image tensor shape: "
            f"expected {expected_shape}, received {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ImagePreprocessingError(
            "Image tensor contains NaN or infinite values."
        )

    return array


def prepare_reconstruction_input(image: Image.Image) -> np.ndarray:
    """
    Prepare input for Notebook 1/2/3 reconstruction-based models.

    Contract:
        shape : (1, 128, 128, 3)
        dtype : float32
        range : [0, 1]
    """
    array = _resize_rgb(image, RECONSTRUCTION_SIZE)
    array = array / np.float32(255.0)

    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ImagePreprocessingError(
            "Normalized reconstruction input must remain inside [0, 1]."
        )

    return np.expand_dims(array, axis=0).astype(
        np.float32,
        copy=False,
    )


def prepare_svm_input(image: Image.Image) -> np.ndarray:
    """
    Prepare input for the Notebook 1 PCA + One-Class SVM baseline.

    The PCA artifact expects the same normalized 128x128 RGB representation,
    flattened to one feature vector.
    """
    batch = prepare_reconstruction_input(image)

    return batch.reshape(
        batch.shape[0],
        -1,
    )


def prepare_mobilenet_input(image: Image.Image) -> np.ndarray:
    """
    Prepare input for the Notebook 4 MobileNetV2 classifier.

    Contract:
        shape : (1, 224, 224, 3)
        dtype : float32
        transform : MobileNetV2 preprocess_input
    """
    array = _resize_rgb(image, MOBILENET_SIZE)

    batch = np.expand_dims(
        array,
        axis=0,
    ).astype(np.float32, copy=False)

    batch = preprocess_input(batch)

    if batch.shape != (1, 224, 224, 3):
        raise ImagePreprocessingError(
            f"Unexpected MobileNetV2 input shape: {batch.shape}."
        )

    if not np.isfinite(batch).all():
        raise ImagePreprocessingError(
            "MobileNetV2 input contains NaN or infinite values."
        )

    return batch