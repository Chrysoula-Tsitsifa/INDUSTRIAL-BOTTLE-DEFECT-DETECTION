from __future__ import annotations

import streamlit as st

from app.core.inference import InferenceResult


def render_app_header() -> None:
    """Render the main application identity."""
    st.markdown(
        """
        <div class="app-header">
            <h1>Industrial Bottle Defect Detection</h1>
            <p>
                Multi-model visual inspection pipeline with calibrated
                MobileNetV2 inference and Grad-CAM explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_context(model_key: str) -> None:
    """Explain the scientific role of the selected inference engine."""
    if model_key == "mobilenet_v2":
        st.markdown(
            """
            <div class="deployment-note">
                <strong>Deployment candidate:</strong>
                MobileNetV2 is the selected supervised classifier with
                Platt-calibrated P(Good) and a calibration-only operational
                threshold. The exported TFLite model remains under Quality
                Hold; this application uses the validated Keras artifact.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="research-note">
                <strong>Research comparison engine:</strong>
                this model is retained as thesis evidence and comparative
                analysis. It is not presented as an equivalent production
                candidate to the calibrated MobileNetV2 classifier.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result(
    result: InferenceResult,
) -> None:
    """Render one inference result without altering its decision contract."""
    if result.review_required:
        css_class = "status-review"
        status_text = "MANUAL REVIEW"
    elif result.is_defective:
        css_class = "status-defective"
        status_text = "DEFECTIVE"
    else:
        css_class = "status-good"
        status_text = "GOOD"

    st.markdown(
        f"""
        <div class="status-card {css_class}">
            <div class="metric-label">Inspection Decision</div>
            <div class="metric-value">{status_text}</div>
            <div style="margin-top:0.35rem; color:#6b7280;">
                {result.model_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_column, threshold_column, class_column = st.columns(3)

    with score_column:
        st.metric(
            label=result.score_name,
            value=f"{result.score:.4f}",
        )

    with threshold_column:
        st.metric(
            label="Decision Threshold",
            value=f"{result.threshold:.4f}",
        )

    with class_column:
        st.metric(
            label="Predicted Class",
            value=result.label,
        )

    if result.calibrated_probability_good is not None:
        st.caption(
            "Operational probability: "
            f"P(Good) = {result.calibrated_probability_good:.4f}"
        )

    if result.raw_probability_good is not None:
        st.caption(
            "Raw neural output before Platt calibration: "
            f"P(Good) = {result.raw_probability_good:.4f}"
        )

    if result.ssim_similarity is not None:
        st.caption(
            "Structural similarity: "
            f"SSIM = {result.ssim_similarity:.4f} "
            f"(anomaly score = {result.score:.4f})"
        )

    if result.review_required:
        st.warning(
            "The calibrated probability falls inside the operational "
            "review band. The model decision is preserved, but manual "
            "inspection is recommended."
        )


def render_xai_limitations() -> None:
    """Display the scientific interpretation boundary for Grad-CAM."""
    st.info(
        "Grad-CAM highlights spatial regions that contributed strongly "
        "to the selected neural class score. It is post-hoc diagnostic "
        "evidence and does not prove pixel-accurate physical defect "
        "localization or causal correctness."
    )


def render_deployment_scope() -> None:
    """Render the deployment-candidate limitation explicitly."""
    st.caption(
        "Deployment scope: host-side candidate only. "
        "Target-device benchmarking and independent operational "
        "validation remain required."
    )