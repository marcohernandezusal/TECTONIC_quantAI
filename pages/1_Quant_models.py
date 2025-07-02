# streamlit_front.py
"""Streamlit UI adapted to the FastAPI backend (v0.2)

Usage:
    streamlit run streamlit_front.py

Env / Secrets:
    API_URL   Base URL of the FastAPI service (default: http://localhost:8000)

"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
import streamlit as st

###############################################################################
# Config                                                                      #
###############################################################################
import os                                                       # ← nuevo

# URL del backend FastAPI:
API_URL: str = os.getenv("API_URL", "http://localhost:8000")    # ← cambio clave

st.set_page_config(
    page_title="Corrosion Prediction – quantAI",
    layout="centered",
)

###############################################################################
# Helpers                                                                     #
###############################################################################

def get_session_id(client: httpx.Client) -> str:
    """Fetch or reuse a session UUID stored in st.session_state."""
    if "session_id" not in st.session_state:
        resp = client.post("/sessions", timeout=10)
        resp.raise_for_status()
        st.session_state["session_id"] = resp.json()["id"]
    return st.session_state["session_id"]


def post_file(
    endpoint: str,
    file,  # streamlit UploadedFile
    headers: dict,
    params: dict,
    client: httpx.Client,
):
    files = {"data_file": (file.name, file, file.type)}
    resp = client.post(endpoint, headers=headers, params=params, files=files, timeout=None)
    resp.raise_for_status()
    return resp.json()


def fetch_binary(endpoint: str, headers: dict[str, str], client: httpx.Client) -> bytes:
    resp = client.get(endpoint, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.content

###############################################################################
# UI                                                                          #
###############################################################################

st.title("Corrosion Prediction for Underwater Cultural Heritage")

uploaded_file = st.file_uploader(
    label="1. Upload Environmental Data (.csv or .xlsx)",
    type=["csv", "xlsx"],
)

model_type = st.selectbox(
    label="2. Choose Model Type", options=["supervised", "selftraining"], key="model_type"
)

retrain_flag = False
if model_type == "selftraining":
    retrain_flag = st.checkbox("Enable continuous learning with uploaded data", key="retrain")

if st.button("Run Model"):
    if not uploaded_file:
        st.warning("Please upload a valid data file before running the model.")
        st.stop()

    with httpx.Client(base_url=API_URL) as client:
        session_id = get_session_id(client)
        headers = {"X-Session-ID": session_id}

        # Decide endpoint and query params
        if model_type == "supervised":
            endpoint = "/predict"
            params = {"model_type": "supervised"}
        else:  # selftraining
            endpoint = "/selftraining"
            params = {"retrain": str(retrain_flag).lower()}  # 'true' | 'false'

        with st.spinner("Running inference on backend – please wait…"):
            try:
                result = post_file(endpoint, uploaded_file, headers, params, client)
            except httpx.HTTPError as ex:
                st.error(f"Backend error: {ex.response.text if ex.response else ex}")
                st.stop()

        st.success("Prediction complete.")

        # Download CSV
        csv_bytes = fetch_binary(f"/predictions/{result['id']}/csv", headers, client)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_bytes,
            file_name=Path(result["output_csv"]).name,
            mime="text/csv",
        )

        # Show plot
        img_bytes = fetch_binary(f"/predictions/{result['id']}/plot", headers, client)
        st.image(img_bytes, caption="Corrosion Prediction Plot", use_container_width=True)

        # Optional: display log if backend exposes it (future endpoint)
        if result.get("log_path"):
            st.info("Inference log stored server‑side; endpoint not yet exposed.")
