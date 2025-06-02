import streamlit as st
from deployment.predict_model import run_prediction
from deployment.retrain_selftraining import run_selftraining
import tempfile
import os

# --- Title ---
st.title("Corrosion Prediction for Underwater Cultural Heritage")

# --- File upload ---
st.subheader("1. Upload Environmental Data")
uploaded_file = st.file_uploader(
    label="Select your environmental data file (.csv or .xlsx)", 
    type=["csv", "xlsx"]
)

# --- Model selection ---
st.subheader("2. Choose Model Type")
model_type = st.selectbox(
    label="Prediction model", 
    options=["supervised", "selftraining"]
)

# --- Retraining option (conditional) ---
retrain = False
if model_type == "selftraining":
    st.subheader("3. Optional: Retrain Model")
    retrain = st.checkbox("Enable continuous learning with uploaded data")

# --- Run button ---
st.subheader("4. Execute Prediction")
if st.button("Run Model"):
    if not uploaded_file:
        st.warning("Please upload a valid data file before running the model.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # --- Run backend logic ---
    if retrain and model_type == "selftraining":
        st.info("Starting self-training with uploaded data...")
        messages = []
        def stream_msg(msg):
            messages.append(msg)
            st.write(msg)

        result = run_selftraining(input_path, st_logger=stream_msg)
    else:
        st.info(f"Running prediction using the {model_type} model...")
        result = run_prediction(model_type, input_path)

    # --- Display results ---
    st.success("Prediction complete.")
    st.download_button(
        label="Download Predictions (CSV)", 
        data=open(result["csv"], "rb"), 
        file_name=os.path.basename(result["csv"])
    )
    st.image(result["plot"], caption="Corrosion Prediction Plot", use_container_width=True)

    if model_type == "selftraining" and retrain and result.get("train_plot"):
        st.image(result["train_plot"], caption="Pseudo-Labeling Progression", use_container_width=True)

    try:
        with open(result["log"], "rb") as log_file:
            st.download_button(
                label="Download Inference Log",
                data=log_file,
                file_name=os.path.basename(result["log"])
            )
    except FileNotFoundError:
        st.warning("Log file not found.")
