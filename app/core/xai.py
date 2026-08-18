from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import colormaps

from app.core.inference import (
    RuntimeModels,
    load_runtime_models,
    predict_mobilenet_v2,
)
from app.core.preprocessing import prepare_mobilenet_input
from app.core.validation import validate_image_for_inference


class XAIError(RuntimeError):
    """Raised when Grad-CAM evidence cannot be generated safely."""


TARGET_LAYER: Final[str] = "out_relu"


@dataclass(frozen=True)
class GradCAMModels:
    """Verified feature/head decomposition used for Grad-CAM."""

    conv_model: Any
    classifier_model: Any
    target_layer: str
    target_signal: str


@dataclass(frozen=True)
class GradCAMResult:
    """Post-hoc Grad-CAM evidence for one MobileNetV2 decision."""

    heatmap: np.ndarray
    overlay: Image.Image
    target_class: int
    target_label: str
    target_layer: str
    target_signal: str


def _build_gradcam_models(
    model: Any,
    requested_layer: str = TARGET_LAYER,
) -> GradCAMModels:
    """
    Reproduce Notebook 4's verified Grad-CAM feature/head decomposition.

    The selected classifier contains the nested MobileNetV2 feature extractor
    as its first layer. Grad-CAM operates on that spatial feature output.
    """
    if not model.layers or not isinstance(
        model.layers[0],
        tf.keras.Model,
    ):
        raise XAIError(
            "Expected the classifier to contain a nested convolutional "
            "feature extractor as its first layer."
        )

    base_model = model.layers[0]

    base_output_shape = tf.TensorShape(
        base_model.output_shape
    )

    if base_output_shape.rank != 4:
        raise XAIError(
            "MobileNetV2 feature output must be spatial rank-4. "
            f"Received: {base_output_shape}."
        )

    output_layer = base_model.layers[-1]

    target_name = output_layer.name

    target_signal = (
        "SUCCESS"
        if requested_layer == target_name
        else "FALLBACK"
    )

    if requested_layer != target_name:
        try:
            requested = base_model.get_layer(
                requested_layer
            )

            if requested is output_layer:
                target_name = requested.name
                target_signal = "SUCCESS"

        except ValueError:
            pass

    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.output,
        name="gradcam_feature_extractor",
    )

    head_input = tf.keras.Input(
        shape=base_output_shape[1:],
        name="gradcam_classifier_input",
    )

    x = head_input

    for layer in model.layers[1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(
        inputs=head_input,
        outputs=x,
        name="gradcam_classifier_head",
    )

    probe = tf.zeros(
        (1, 224, 224, 3),
        dtype=tf.float32,
    )

    full_prediction = np.asarray(
        model(
            probe,
            training=False,
        )
    ).reshape(-1)

    split_prediction = np.asarray(
        classifier_model(
            conv_model(
                probe,
                training=False,
            ),
            training=False,
        )
    ).reshape(-1)

    if (
        full_prediction.size != 1
        or split_prediction.size != 1
        or not np.allclose(
            full_prediction,
            split_prediction,
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise XAIError(
            "Grad-CAM feature/head decomposition failed "
            "model-output parity verification."
        )

    return GradCAMModels(
        conv_model=conv_model,
        classifier_model=classifier_model,
        target_layer=target_name,
        target_signal=target_signal,
    )


@lru_cache(maxsize=1)
def load_gradcam_models() -> GradCAMModels:
    """Load and cache the verified Notebook 4 Grad-CAM graph."""
    runtime = load_runtime_models()

    return _build_gradcam_models(
        runtime.mobilenet,
    )


def _make_gradcam_heatmap(
    image_batch: np.ndarray,
    models: GradCAMModels,
    target_class: int,
) -> np.ndarray:
    """
    Generate Notebook 4's binary class-symmetric Grad-CAM heatmap.

    target_class:
        0 -> Defective
        1 -> Good
    """
    if target_class not in (0, 1):
        raise XAIError(
            f"Grad-CAM target class must be 0 or 1, received {target_class}."
        )

    with tf.GradientTape() as tape:
        activations = models.conv_model(
            image_batch,
            training=False,
        )

        tape.watch(activations)

        predictions = models.classifier_model(
            activations,
            training=False,
        )

        if (
            predictions.shape.rank != 2
            or predictions.shape[-1] != 1
        ):
            raise XAIError(
                "Expected sigmoid classifier output shape (N, 1), "
                f"received {predictions.shape}."
            )

        p_good = tf.clip_by_value(
            predictions[:, 0],
            1e-6,
            1.0 - 1e-6,
        )

        logit_good = (
            tf.math.log(p_good)
            - tf.math.log1p(-p_good)
        )

        target_score = (
            logit_good
            if target_class == 1
            else -logit_good
        )

    gradients = tape.gradient(
        target_score,
        activations,
    )

    if gradients is None:
        raise XAIError(
            "Grad-CAM gradient is disconnected from the spatial features."
        )

    activations_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                activations
            )
        ).numpy()
    )

    gradients_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                gradients
            )
        ).numpy()
    )

    if not activations_finite or not gradients_finite:
        raise XAIError(
            "Grad-CAM produced non-finite activations or gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    heatmap = tf.reduce_sum(
        activations[0] * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.nn.relu(
        heatmap
    )

    maximum = float(
        tf.reduce_max(
            heatmap
        ).numpy()
    )

    total_energy = float(
        tf.reduce_sum(
            heatmap
        ).numpy()
    )

    if (
        not np.isfinite(maximum)
        or not np.isfinite(total_energy)
        or maximum <= 1e-10
        or total_energy <= 1e-10
    ):
        raise XAIError(
            "Grad-CAM produced an empty activation map; "
            "no artificial point of interest will be generated."
        )

    heatmap = heatmap / maximum

    result = np.asarray(
        heatmap.numpy(),
        dtype=np.float32,
    )

    if (
        result.ndim != 2
        or not np.isfinite(result).all()
    ):
        raise XAIError(
            "Grad-CAM heatmap failed dimensional or finite-value validation."
        )

    return result


def _build_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Render the Grad-CAM map over the original RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise XAIError(
            "Grad-CAM overlay alpha must remain inside [0, 1]."
        )

    original = image.convert("RGB")

    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap,
            0.0,
            1.0,
        )
        * 255.0
    )

    resized_heatmap = Image.fromarray(
        heatmap_uint8,
        mode="L",
    ).resize(
        original.size,
        resample=Image.Resampling.BILINEAR,
    )

    normalized_heatmap = (
        np.asarray(
            resized_heatmap,
            dtype=np.float32,
        )
        / 255.0
    )

    colored = colormaps["jet"](
        normalized_heatmap
    )[..., :3]

    colored_uint8 = np.uint8(
        np.clip(
            colored,
            0.0,
            1.0,
        )
        * 255.0
    )

    heatmap_image = Image.fromarray(
        colored_uint8,
        mode="RGB",
    )

    return Image.blend(
        original,
        heatmap_image,
        alpha=alpha,
    )


def generate_gradcam(
    image: Image.Image,
    runtime: RuntimeModels | None = None,
) -> GradCAMResult:
    """
    Generate Grad-CAM for the calibrated Notebook 4 decision.

    The calibrated classifier decision determines which binary class is
    explained. Grad-CAM remains post-hoc diagnostic evidence and does not
    modify the operational prediction.
    """
    validate_image_for_inference(
        image
    )

    runtime = (
        runtime
        if runtime is not None
        else load_runtime_models()
    )

    prediction = predict_mobilenet_v2(
        image,
        runtime,
    )

    target_class = (
        0
        if prediction.is_defective
        else 1
    )

    target_label = (
        "Defective"
        if target_class == 0
        else "Good"
    )

    image_batch = prepare_mobilenet_input(
        image
    )

    models = load_gradcam_models()

    heatmap = _make_gradcam_heatmap(
        image_batch,
        models,
        target_class,
    )

    overlay = _build_overlay(
        image,
        heatmap,
    )

    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        target_class=target_class,
        target_label=target_label,
        target_layer=models.target_layer,
        target_signal=models.target_signal,
    )