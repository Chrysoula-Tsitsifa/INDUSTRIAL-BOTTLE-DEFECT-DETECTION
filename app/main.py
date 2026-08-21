from __future__ import annotations

import streamlit as st

from app.core.inference import (
    MODEL_OPTIONS,
    InferenceError,
    run_inference,
)
from app.core.preprocessing import (
    ImagePreprocessingError,
    decode_image,
)
from app.core.validation import InputValidationError
from app.core.xai import (
    XAIError,
    generate_gradcam,
)
from app.ui.components import (
    render_app_header,
    render_deployment_scope,
    render_model_context,
    render_prediction_result,
    render_xai_limitations,
)
from app.ui.styles import apply_app_styles


def main() -> None:
    """Run the Streamlit industrial inspection application."""

    st.set_page_config(
        page_title="Industrial Bottle Defect Detection",
        page_icon="🔬",
        layout="wide",
    )

    apply_app_styles()
    render_app_header()

    upload_column, control_column = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    image = None

    with upload_column:
        st.markdown(
            '<div class="section-title">Inspection Image</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload a bottle image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "bmp",
            ],
            help=(
                "Upload one image for technical validation "
                "and model inference."
            ),
        )

        if uploaded_file is not None:
            try:
                image = decode_image(
                    uploaded_file.getvalue()
                )

                st.image(
                    image,
                    caption=(
                        f"Uploaded image - "
                        f"{image.width} x {image.height}px"
                    ),
                )

            except ImagePreprocessingError as exc:
                st.error(
                    f"Image decoding failed: {exc}"
                )

    with control_column:
        st.markdown(
            '<div class="section-title">Inference Engine</div>',
            unsafe_allow_html=True,
        )

        model_keys = list(
            MODEL_OPTIONS.keys()
        )

        model_key = st.selectbox(
            "Select model",
            options=model_keys,
            index=model_keys.index(
                "mobilenet_v2"
            ),
            format_func=lambda key: MODEL_OPTIONS[key],
        )

        render_model_context(
            model_key
        )

        generate_xai = False

        if model_key == "mobilenet_v2":
            generate_xai = st.checkbox(
                "Generate Grad-CAM explanation",
                value=True,
                help=(
                    "Generate post-hoc spatial evidence for "
                    "the MobileNetV2 decision."
                ),
            )

        run_requested = st.button(
            "Run inspection",
            type="primary",
        )

    if not run_requested:
        if model_key == "mobilenet_v2":
            render_deployment_scope()
        return

    if image is None:
        st.warning(
            "Upload a valid image before running inspection."
        )

        if model_key == "mobilenet_v2":
            render_deployment_scope()

        return

    st.markdown(
        '<div class="section-title">Inspection Result</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(
            "Running validated inference pipeline..."
        ):
            result = run_inference(
                image,
                model_key,
            )

        render_prediction_result(
            result
        )

    except (
        ImagePreprocessingError,
        InputValidationError,
        InferenceError,
    ) as exc:
        st.error(
            f"Inference could not be completed: {exc}"
        )

        if model_key == "mobilenet_v2":
            render_deployment_scope()

        return

    if (
        model_key == "mobilenet_v2"
        and generate_xai
    ):
        st.markdown(
            '<div class="section-title">'
            'Grad-CAM Explainability'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            with st.spinner(
                "Generating Grad-CAM evidence..."
            ):
                xai_result = generate_gradcam(
                    image
                )

            original_column, xai_column = st.columns(
                2,
                gap="large",
            )

            with original_column:
                st.image(
                    image,
                    caption="Original image",
                )

            with xai_column:
                st.image(
                    xai_result.overlay,
                    caption=(
                        "Grad-CAM overlay - "
                        f"target={xai_result.target_label}"
                    ),
                )

            st.caption(
                "Grad-CAM target layer: "
                f"{xai_result.target_layer} "
                f"({xai_result.target_signal})"
            )

            render_xai_limitations()

        except XAIError as exc:
            st.warning(
                "The classifier decision remains valid, "
                "but Grad-CAM evidence could not be generated "
                f"for this image: {exc}"
            )

    if model_key == "mobilenet_v2":
        render_deployment_scope()


if __name__ == "__main__":
    main()