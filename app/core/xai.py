from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image

from app.core.inference import (
    RuntimeModels,
    load_runtime_models,
    predict_mobilenet_v2,
)
from app.core.preprocessing import prepare_mobilenet_input
from app.core.validation import validate_image_for_inference


class XAIError(RuntimeError):
    """Raised when XAI evidence cannot be generated safely."""


TARGET_LAYER: Final[str] = "out_relu"

XAI_CANDIDATE_LAYERS: Final[tuple[str, ...]] = (
    "block_5_add",
    "block_12_add",
    "out_relu",
)


@dataclass(frozen=True)
class GradCAMModels:
    """Verified graph decomposition used for layer-targeted Grad-CAM."""

    conv_model: Any
    gradient_model: Any
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
    Build a verified layer-targeted Grad-CAM graph.

    The classifier contains MobileNetV2 as its first nested model. The
    requested intermediate activation is exposed together with the final
    MobileNetV2 feature tensor, while the original classifier head is
    reconstructed unchanged.

    This allows gradients of the real classifier score to be measured with
    respect to 28x28, 14x14 or 7x7 intermediate feature maps without changing
    model weights or the operational prediction.
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

    try:
        requested = base_model.get_layer(
            requested_layer
        )
    except ValueError as exc:
        raise XAIError(
            f"Requested Grad-CAM layer '{requested_layer}' "
            "does not exist in the MobileNetV2 feature extractor."
        ) from exc

    requested_shape = tf.TensorShape(
        requested.output.shape
    )

    if requested_shape.rank != 4:
        raise XAIError(
            f"Requested Grad-CAM layer '{requested_layer}' "
            "must expose a spatial rank-4 tensor. "
            f"Received: {requested_shape}."
        )

    spatial_height = requested_shape[1]
    spatial_width = requested_shape[2]
    channels = requested_shape[3]

    if (
        spatial_height is None
        or spatial_width is None
        or channels is None
        or spatial_height <= 0
        or spatial_width <= 0
        or channels <= 0
    ):
        raise XAIError(
            f"Requested Grad-CAM layer '{requested_layer}' "
            f"has an invalid spatial output shape: {requested_shape}."
        )

    # Public inspection model: requested activation only.
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=requested.output,
        name=f"gradcam_activation_{requested_layer}",
    )

    # Gradient graph:
    # input -> requested intermediate activation
    #       -> final MobileNetV2 output
    #
    # Returning both tensors from the same forward graph preserves the
    # differentiable path from the classifier score back to the requested
    # intermediate layer.
    gradient_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            requested.output,
            base_model.output,
        ],
        name=f"gradcam_gradient_graph_{requested_layer}",
    )

    # Rebuild only the outer classifier head.
    # The actual trained layer objects and weights are reused unchanged.
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

    # Verify that splitting the model does not alter its prediction.
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

    (
        probe_target_activation,
        probe_base_output,
    ) = gradient_model(
        probe,
        training=False,
    )

    split_prediction = np.asarray(
        classifier_model(
            probe_base_output,
            training=False,
        )
    ).reshape(-1)

    inspected_activation = np.asarray(
        conv_model(
            probe,
            training=False,
        )
    )

    gradient_activation = np.asarray(
        probe_target_activation
    )

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

    if (
        inspected_activation.shape
        != gradient_activation.shape
        or not np.allclose(
            inspected_activation,
            gradient_activation,
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise XAIError(
            "Grad-CAM target activation failed graph parity verification."
        )

    return GradCAMModels(
        conv_model=conv_model,
        gradient_model=gradient_model,
        classifier_model=classifier_model,
        target_layer=requested.name,
        target_signal="SUCCESS",
    )


@lru_cache(maxsize=8)
def load_gradcam_models(
    requested_layer: str = TARGET_LAYER,
) -> GradCAMModels:
    """Load and cache a verified Grad-CAM graph for one target layer."""
    runtime = load_runtime_models()

    return _build_gradcam_models(
        runtime.mobilenet,
        requested_layer=requested_layer,
    )


def _make_gradcam_heatmap(
    image_batch: np.ndarray,
    models: GradCAMModels,
    target_class: int,
) -> np.ndarray:
    """
    Generate binary class-symmetric Grad-CAM evidence.

    target_class:
        0 -> Defective
        1 -> Good
    """
    if target_class not in (0, 1):
        raise XAIError(
            f"Grad-CAM target class must be 0 or 1, "
            f"received {target_class}."
        )

    with tf.GradientTape() as tape:
        (
            target_activations,
            base_features,
        ) = models.gradient_model(
            image_batch,
            training=False,
        )

        tape.watch(
            target_activations
        )

        predictions = models.classifier_model(
            base_features,
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
        target_activations,
    )

    if gradients is None:
        raise XAIError(
            "Grad-CAM gradient is disconnected from "
            f"target layer '{models.target_layer}'."
        )

    activations_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                target_activations
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

    if (
        not activations_finite
        or not gradients_finite
    ):
        raise XAIError(
            "Grad-CAM produced non-finite activations or gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    heatmap = tf.reduce_sum(
        target_activations[0]
        * pooled_gradients,
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
            "Grad-CAM produced an empty activation map "
            f"for target layer '{models.target_layer}'; "
            "no artificial point of interest will be generated."
        )

    heatmap = (
        heatmap
        / maximum
    )

    result = np.asarray(
        heatmap.numpy(),
        dtype=np.float32,
    )

    if (
        result.ndim != 2
        or not np.isfinite(
            result
        ).all()
    ):
        raise XAIError(
            "Grad-CAM heatmap failed dimensional "
            "or finite-value validation."
        )

    return result

def _make_gradcam_pp_heatmap(
    image_batch: np.ndarray,
    models: GradCAMModels,
    target_class: int,
) -> np.ndarray:
    """
    Generate class-symmetric Grad-CAM++ evidence.

    Grad-CAM++ uses higher-order derivatives to derive spatially aware
    channel weights. The explained class is expressed through the same
    binary Good/Defective log-odds convention used by Grad-CAM.

    target_class:
        0 -> Defective
        1 -> Good
    """
    if target_class not in (0, 1):
        raise XAIError(
            f"Grad-CAM++ target class must be 0 or 1, "
            f"received {target_class}."
        )

    with tf.GradientTape() as third_tape:
        with tf.GradientTape() as second_tape:
            with tf.GradientTape() as first_tape:
                (
                    target_activations,
                    base_features,
                ) = models.gradient_model(
                    image_batch,
                    training=False,
                )

                first_tape.watch(
                    target_activations
                )
                second_tape.watch(
                    target_activations
                )
                third_tape.watch(
                    target_activations
                )

                predictions = models.classifier_model(
                    base_features,
                    training=False,
                )

                if (
                    predictions.shape.rank != 2
                    or predictions.shape[-1] != 1
                ):
                    raise XAIError(
                        "Expected sigmoid classifier output "
                        "shape (N, 1), "
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

                signed_logit = (
                    logit_good
                    if target_class == 1
                    else -logit_good
                )

                # Grad-CAM++ requires non-zero higher-order derivatives.
                # Exponentiating the class logit is monotonic, therefore
                # it preserves the explained class ordering while yielding
                # the higher-order derivative structure required by the
                # Grad-CAM++ weighting rule.
                target_score = tf.exp(
                    tf.clip_by_value(
                        signed_logit,
                        -20.0,
                        20.0,
                    )
                )

            first_gradients = first_tape.gradient(
                target_score,
                target_activations,
            )

        if first_gradients is None:
            raise XAIError(
                "Grad-CAM++ first-order gradient is disconnected "
                f"from target layer '{models.target_layer}'."
            )

        second_gradients = second_tape.gradient(
            first_gradients,
            target_activations,
        )

    if second_gradients is None:
        raise XAIError(
            "Grad-CAM++ second-order gradient is disconnected "
            f"from target layer '{models.target_layer}'."
        )

    third_gradients = third_tape.gradient(
        second_gradients,
        target_activations,
    )

    if third_gradients is None:
        raise XAIError(
            "Grad-CAM++ third-order gradient is disconnected "
            f"from target layer '{models.target_layer}'."
        )

    tensors = (
        target_activations,
        first_gradients,
        second_gradients,
        third_gradients,
    )

    if not all(
        bool(
            tf.reduce_all(
                tf.math.is_finite(tensor)
            ).numpy()
        )
        for tensor in tensors
    ):
        raise XAIError(
            "Grad-CAM++ produced non-finite activations "
            "or derivatives."
        )

    activation_sum = tf.reduce_sum(
        target_activations,
        axis=(1, 2),
        keepdims=True,
    )

    alpha_numerator = second_gradients

    alpha_denominator = (
        2.0 * second_gradients
        + third_gradients * activation_sum
    )

    safe_denominator = tf.where(
        tf.abs(alpha_denominator) > 1e-12,
        alpha_denominator,
        tf.ones_like(alpha_denominator),
    )

    alphas = (
        alpha_numerator
        / safe_denominator
    )

    alphas = tf.where(
        tf.abs(alpha_denominator) > 1e-12,
        alphas,
        tf.zeros_like(alphas),
    )

    positive_gradients = tf.nn.relu(
        first_gradients
    )

    channel_weights = tf.reduce_sum(
        alphas * positive_gradients,
        axis=(1, 2),
    )

    heatmap = tf.reduce_sum(
        target_activations[0]
        * channel_weights[0],
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
            "Grad-CAM++ produced an empty activation map "
            f"for target layer '{models.target_layer}'."
        )

    heatmap = (
        heatmap
        / maximum
    )

    result = np.asarray(
        heatmap.numpy(),
        dtype=np.float32,
    )

    if (
        result.ndim != 2
        or not np.isfinite(result).all()
    ):
        raise XAIError(
            "Grad-CAM++ heatmap failed dimensional "
            "or finite-value validation."
        )

    return result

def _build_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Render a Grad-CAM heatmap over the original RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise XAIError(
            "Grad-CAM overlay alpha must remain inside [0, 1]."
        )

    original = image.convert(
        "RGB"
    )

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
    target_layer: str = TARGET_LAYER,
) -> GradCAMResult:
    """
    Generate Grad-CAM for the calibrated Notebook 4 decision.

    The calibrated operational decision determines which binary class is
    explained. The target layer controls XAI spatial resolution only and does
    not alter the classifier, calibration, threshold or operational decision.
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

    if runtime is load_runtime_models():
        models = load_gradcam_models(
            target_layer
        )
    else:
        models = _build_gradcam_models(
            runtime.mobilenet,
            requested_layer=target_layer,
        )

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