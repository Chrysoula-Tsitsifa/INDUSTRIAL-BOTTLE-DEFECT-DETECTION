from __future__ import annotations

from dataclasses import dataclass
from email.mime import image
from functools import lru_cache
from pathlib import Path
from typing import Any, Final
from app.core.validation import validate_image_for_inference

import joblib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from tensorflow.keras.models import load_model

from app.core.artifacts import verify_artifact_integrity
from app.core.contracts import ARTIFACT_ROOT, load_all_contracts
from app.core.preprocessing import (
    prepare_mobilenet_input,
    prepare_reconstruction_input,
    prepare_svm_input,
)


class InferenceError(RuntimeError):
    """Raised when a deployment inference contract cannot be satisfied."""


MODEL_OPTIONS: Final[dict[str, str]] = {
    "svm_pca": "Notebook 1 — PCA + One-Class SVM",
    "simple_ae": "Notebook 1 — Simple Autoencoder",
    "optimized_ae": "Notebook 2 — Tuned Convolutional Autoencoder",
    "ssim": "Notebook 3 — SSIM Structural Reconstruction",
    "mobilenet_v2": "Notebook 4 — MobileNetV2 Calibrated Classifier",
}


@dataclass(frozen=True)
class InferenceResult:
    """Normalized result returned by every inference engine."""

    model_key: str
    model_name: str
    label: str
    is_defective: bool
    score: float
    threshold: float
    score_name: str
    review_required: bool = False
    raw_probability_good: float | None = None
    calibrated_probability_good: float | None = None
    ssim_similarity: float | None = None


@dataclass(frozen=True)
class RuntimeModels:
    """Frozen deployment artifacts loaded from the canonical package."""

    pca: Any
    svm: Any
    simple_ae: Any
    tuned_ae: Any
    mobilenet: Any
    contracts: dict[str, dict[str, Any]]


def _artifact_path(
    contract: dict[str, Any],
    artifact_key: str,
) -> Path:
    artifacts = contract.get("artifacts")

    if not isinstance(artifacts, dict):
        raise InferenceError(
            "Contract does not contain a valid artifacts mapping."
        )

    relative_path = artifacts.get(artifact_key)

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise InferenceError(
            f"Artifact key is missing from contract: {artifact_key}"
        )

    candidate = (ARTIFACT_ROOT / relative_path).resolve()
    root = ARTIFACT_ROOT.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InferenceError(
            f"Artifact path escapes canonical artifact root: {relative_path}"
        ) from exc

    if not candidate.is_file():
        raise InferenceError(
            f"Canonical artifact does not exist: {candidate}"
        )

    return candidate


@lru_cache(maxsize=1)
def load_runtime_models() -> RuntimeModels:
    """
    Verify the canonical package, validate contracts and load frozen models.

    Loading is cached so Streamlit reruns do not repeatedly deserialize
    large model artifacts.
    """
    verify_artifact_integrity()

    contracts = load_all_contracts()

    notebook1 = contracts["notebook1"]
    notebook2 = contracts["notebook2"]
    notebook4 = contracts["notebook4"]

    try:
        pca = joblib.load(
            _artifact_path(
                notebook1,
                "pca_model",
            )
        )

        svm = joblib.load(
            _artifact_path(
                notebook1,
                "svm_model",
            )
        )

        simple_ae = load_model(
            _artifact_path(
                notebook1,
                "simple_ae_model",
            ),
            compile=False,
        )

        tuned_ae = load_model(
            _artifact_path(
                notebook2,
                "tuned_model",
            ),
            compile=False,
        )

        mobilenet = load_model(
            _artifact_path(
                notebook4,
                "keras_model",
            ),
            compile=False,
        )

    except Exception as exc:
        raise InferenceError(
            f"Failed to load canonical model artifacts: {exc}"
        ) from exc

    return RuntimeModels(
        pca=pca,
        svm=svm,
        simple_ae=simple_ae,
        tuned_ae=tuned_ae,
        mobilenet=mobilenet,
        contracts=contracts,
    )


def _predict_reconstruction(
    model: Any,
    batch: np.ndarray,
) -> np.ndarray:
    """Run one autoencoder reconstruction with strict shape validation."""
    try:
        reconstruction = model.predict(
            batch,
            verbose=0,
        )
    except Exception as exc:
        raise InferenceError(
            f"Autoencoder inference failed: {exc}"
        ) from exc

    reconstruction = np.asarray(
        reconstruction,
        dtype=np.float32,
    )

    if reconstruction.shape != batch.shape:
        raise InferenceError(
            "Autoencoder reconstruction shape mismatch: "
            f"input={batch.shape}, output={reconstruction.shape}."
        )

    if not np.isfinite(reconstruction).all():
        raise InferenceError(
            "Autoencoder produced NaN or infinite values."
        )

    return reconstruction


def _reconstruction_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Return the notebook-consistent per-image reconstruction MSE."""
    if original.shape != reconstructed.shape:
        raise InferenceError(
            "Cannot calculate reconstruction MSE for mismatched tensors."
        )

    score = float(
        np.mean(
            np.square(
                original - reconstructed
            )
        )
    )

    if not np.isfinite(score):
        raise InferenceError(
            "Reconstruction MSE is not finite."
        )

    return score


def _anomaly_result(
    *,
    model_key: str,
    score: float,
    threshold: float,
    score_name: str,
) -> InferenceResult:
    """
    Build a result for anomaly-score models.

    Notebook 1/2/3 contract:
        score > threshold  -> Defective
        score <= threshold -> Good
    """
    if not np.isfinite(score):
        raise InferenceError(
            f"{model_key} produced a non-finite score."
        )

    if not np.isfinite(threshold):
        raise InferenceError(
            f"{model_key} contract contains an invalid threshold."
        )

    is_defective = bool(score > threshold)

    return InferenceResult(
        model_key=model_key,
        model_name=MODEL_OPTIONS[model_key],
        label="Defective" if is_defective else "Good",
        is_defective=is_defective,
        score=float(score),
        threshold=float(threshold),
        score_name=score_name,
    )


def predict_svm_pca(
    image: Image.Image,
    runtime: RuntimeModels,
) -> InferenceResult:
    """Notebook 1 PCA + One-Class SVM inference."""
    features = prepare_svm_input(image)

    try:
        reduced = runtime.pca.transform(features)

        anomaly_score = float(
            -np.asarray(
                runtime.svm.decision_function(reduced)
            ).reshape(-1)[0]
        )
    except Exception as exc:
        raise InferenceError(
            f"PCA + One-Class SVM inference failed: {exc}"
        ) from exc

    threshold = float(
        runtime.contracts["notebook1"]["models"]["svm_pca"]["threshold"]
    )

    return _anomaly_result(
        model_key="svm_pca",
        score=anomaly_score,
        threshold=threshold,
        score_name="SVM anomaly score",
    )


def predict_simple_ae(
    image: Image.Image,
    runtime: RuntimeModels,
) -> InferenceResult:
    """Notebook 1 Simple Autoencoder inference."""
    batch = prepare_reconstruction_input(image)

    reconstruction = _predict_reconstruction(
        runtime.simple_ae,
        batch,
    )

    score = _reconstruction_mse(
        batch,
        reconstruction,
    )

    threshold = float(
        runtime.contracts["notebook1"]["models"]["simple_ae"]["threshold"]
    )

    return _anomaly_result(
        model_key="simple_ae",
        score=score,
        threshold=threshold,
        score_name="Reconstruction MSE",
    )


def predict_optimized_ae(
    image: Image.Image,
    runtime: RuntimeModels,
) -> InferenceResult:
    """Notebook 2 tuned convolutional Autoencoder inference."""
    batch = prepare_reconstruction_input(image)

    reconstruction = _predict_reconstruction(
        runtime.tuned_ae,
        batch,
    )

    score = _reconstruction_mse(
        batch,
        reconstruction,
    )

    threshold = float(
        runtime.contracts["notebook2"]["model_metrics"]["threshold"]
    )

    return _anomaly_result(
        model_key="optimized_ae",
        score=score,
        threshold=threshold,
        score_name="Reconstruction MSE",
    )


def predict_ssim(
    image: Image.Image,
    runtime: RuntimeModels,
) -> InferenceResult:
    """
    Notebook 3 structural inference.

    Notebook 3 owns no separate model artifact. It reuses the frozen
    Notebook 2 tuned autoencoder and changes only the anomaly metric.
    """
    batch = prepare_reconstruction_input(image)

    reconstruction = _predict_reconstruction(
        runtime.tuned_ae,
        batch,
    )

    try:
        similarity = float(
            structural_similarity(
                batch[0],
                reconstruction[0],
                data_range=1.0,
                channel_axis=-1,
            )
        )
    except Exception as exc:
        raise InferenceError(
            f"SSIM calculation failed: {exc}"
        ) from exc

    if not np.isfinite(similarity):
        raise InferenceError(
            "SSIM returned a non-finite similarity score."
        )

    anomaly_score = float(
        1.0 - similarity
    )

    threshold = float(
        runtime.contracts["notebook3"]["structural_metrics"]["threshold"]
    )

    base_result = _anomaly_result(
        model_key="ssim",
        score=anomaly_score,
        threshold=threshold,
        score_name="1 - SSIM anomaly score",
    )

    return InferenceResult(
        model_key=base_result.model_key,
        model_name=base_result.model_name,
        label=base_result.label,
        is_defective=base_result.is_defective,
        score=base_result.score,
        threshold=base_result.threshold,
        score_name=base_result.score_name,
        ssim_similarity=similarity,
    )


def _apply_platt_scaling(
    probability_good: float,
    slope: float,
    intercept: float,
    eps: float = 1e-6,
) -> float:
    """Apply Notebook 4's frozen Platt calibration mapping."""
    if not np.isfinite(probability_good):
        raise InferenceError(
            "Raw MobileNet probability is not finite."
        )

    if probability_good < 0.0 or probability_good > 1.0:
        raise InferenceError(
            "Raw MobileNet probability is outside [0, 1]."
        )

    clipped = float(
        np.clip(
            probability_good,
            eps,
            1.0 - eps,
        )
    )

    logit = float(
        np.log(
            clipped / (1.0 - clipped)
        )
    )

    z = float(
        np.clip(
            slope * logit + intercept,
            -50.0,
            50.0,
        )
    )

    calibrated = float(
        1.0 / (1.0 + np.exp(-z))
    )

    return calibrated


def predict_mobilenet_v2(
    image: Image.Image,
    runtime: RuntimeModels,
) -> InferenceResult:
    """Notebook 4 calibrated MobileNetV2 inference."""
    batch = prepare_mobilenet_input(image)

    try:
        prediction = np.asarray(
            runtime.mobilenet.predict(
                batch,
                verbose=0,
            ),
            dtype=np.float64,
        ).reshape(-1)
    except Exception as exc:
        raise InferenceError(
            f"MobileNetV2 inference failed: {exc}"
        ) from exc

    if prediction.size != 1:
        raise InferenceError(
            "MobileNetV2 must return exactly one sigmoid P(Good) value; "
            f"received shape {prediction.shape}."
        )

    raw_probability_good = float(
        prediction[0]
    )

    calibration = runtime.contracts["notebook4"][
        "probability_calibration"
    ]

    slope = float(
        calibration["slope"]
    )

    intercept = float(
        calibration["intercept"]
    )

    calibrated_probability_good = _apply_platt_scaling(
        raw_probability_good,
        slope,
        intercept,
    )

    decision_contract = runtime.contracts["notebook4"][
        "decision_contract"
    ]

    threshold = float(
        decision_contract["operational_threshold"]
    )

    review_band = float(
        decision_contract["operational_review_band"]
    )

    lower_review = max(
        0.0,
        threshold - review_band,
    )

    upper_review = min(
        1.0,
        threshold + review_band,
    )

    review_required = bool(
        lower_review
        <= calibrated_probability_good
        <= upper_review
    )

    is_defective = bool(
        calibrated_probability_good < threshold
    )

    return InferenceResult(
        model_key="mobilenet_v2",
        model_name=MODEL_OPTIONS["mobilenet_v2"],
        label="Defective" if is_defective else "Good",
        is_defective=is_defective,
        score=calibrated_probability_good,
        threshold=threshold,
        score_name="Calibrated P(Good)",
        review_required=review_required,
        raw_probability_good=raw_probability_good,
        calibrated_probability_good=calibrated_probability_good,
    )


def run_inference(
    image: Image.Image,
    model_key: str,
) -> InferenceResult:
    """Dispatch one image to the selected frozen inference engine."""
    if model_key not in MODEL_OPTIONS:
        valid = ", ".join(
            MODEL_OPTIONS.keys()
        )

        raise InferenceError(
            f"Unknown model key '{model_key}'. Valid options: {valid}"
        )
    
    validate_image_for_inference(image)
    
    runtime = load_runtime_models()

    dispatch = {
        "svm_pca": predict_svm_pca,
        "simple_ae": predict_simple_ae,
        "optimized_ae": predict_optimized_ae,
        "ssim": predict_ssim,
        "mobilenet_v2": predict_mobilenet_v2,
    }

    return dispatch[model_key](
        image,
        runtime,
    )