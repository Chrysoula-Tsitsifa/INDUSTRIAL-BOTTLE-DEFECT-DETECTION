from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


class InputValidationError(ValueError):
    """Raised when an image cannot safely enter the inference pipeline."""


@dataclass(frozen=True)
class ImageValidationResult:
    width: int
    height: int
    channels: int
    mode: str


def validate_image_for_inference(
    image: Image.Image,
) -> ImageValidationResult:
    """
    Validate the technical image contract before model preprocessing.

    This validates image integrity and tensor suitability only.
    It does not claim to determine whether the image actually contains a bottle.
    """
    if not isinstance(image, Image.Image):
        raise InputValidationError(
            "Expected a PIL.Image.Image instance."
        )

    width, height = image.size

    if width <= 0 or height <= 0:
        raise InputValidationError(
            "Image dimensions must be positive."
        )

    if width < 32 or height < 32:
        raise InputValidationError(
            "Image resolution is too small for reliable model inference."
        )

    rgb_image = image.convert("RGB")

    array = np.asarray(
        rgb_image,
        dtype=np.float32,
    )

    if array.ndim != 3 or array.shape[-1] != 3:
        raise InputValidationError(
            f"Expected an RGB image tensor, received shape {array.shape}."
        )

    if not np.isfinite(array).all():
        raise InputValidationError(
            "Image contains NaN or infinite numerical values."
        )

    return ImageValidationResult(
        width=int(width),
        height=int(height),
        channels=3,
        mode="RGB",
    )