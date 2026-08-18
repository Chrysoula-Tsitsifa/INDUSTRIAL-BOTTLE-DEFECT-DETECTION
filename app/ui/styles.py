from __future__ import annotations

import streamlit as st


APP_CSS = """
<style>
    .stApp {
        background-color: #f7f8fa;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-header {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: #111827;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }

    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .app-header p {
        margin: 0.5rem 0 0 0;
        color: #d1d5db;
        font-size: 0.95rem;
    }

    .status-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .status-good {
        border-left: 6px solid #15803d;
    }

    .status-defective {
        border-left: 6px solid #b91c1c;
    }

    .status-review {
        border-left: 6px solid #d97706;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-value {
        color: #111827;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .section-title {
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
    }

    .research-note {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        font-size: 0.9rem;
    }

    .deployment-note {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        font-size: 0.9rem;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.4rem;
    }
</style>
"""


def apply_app_styles() -> None:
    """Inject the application's shared Streamlit CSS."""
    st.markdown(
        APP_CSS,
        unsafe_allow_html=True,
    )