from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the repository root is available for absolute "app.*" imports
# when Streamlit Community Cloud executes app/main.py directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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