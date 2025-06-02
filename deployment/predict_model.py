import os
import sys
import argparse
import joblib
import logging
import pandas as pd
from datetime import datetime
import json

from pipeline.data_loader import read_input_file, check_columns
from pipeline.evaluation import plot_predictions_array
from pipeline.utils import setup_logger, get_latest_model


# ---------- Configuration ----------
MODEL_DIR = "models"
LOG_DIR = "inference_logs"
OUTPUT_DIR = "outputs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_prediction(model_type: str, input_path: str) -> dict:
    """
    Perform inference using the latest saved model of the given type.

    Returns:
        dict with paths to the prediction CSV, plot, and log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log, log_path = setup_logger(model_type.upper(), timestamp)
    log.info(f"--- Starting {model_type.upper()} inference ---")

    try:
        model_path = get_latest_model(model_type)
        model = joblib.load(model_path)
        metadata_path = model_path.replace(".pkl", ".json")

        scaler_path = 'scaler/scaler.pkl'
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                scaler_path = meta.get("scaler_used")
                expected_features = meta.get("features")
                scaler = joblib.load(scaler_path)
        else:
            scaler = joblib.load(scaler_path)
            expected_features = scaler.feature_names_in_        

        df = read_input_file(input_path)
        check_columns(df, expected_features)

        X = df[expected_features].copy()
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Save predictions
        result_df = df.copy()
        result_df["Predicted_Corrosion"] = y_pred
        csv_out = os.path.join(OUTPUT_DIR, f"predictions_{model_type}_{timestamp}.csv")
        result_df.to_csv(csv_out, index=False)

        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, f"prediction_plot_{model_type}_{timestamp}.png")
        plot_predictions_array(y_pred, plot_file)

        log.info(f"Prediction complete. Results saved to: {csv_out}")
        log.info(f"Plot saved to: {plot_file}")
        log.info(f"Inference log saved to: {log_path}")

        return {
            "csv": csv_out,
            "plot": plot_file,
            "log": log_path,
            "timestamp": timestamp
        }

    except Exception as e:
        log.exception(f"Inference failed: {e}")
        raise RuntimeError(f"Inference failed. See log: {log_path}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the latest model.")
    parser.add_argument("--model_type", type=str, choices=["supervised", "selftraining"], required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    try:
        result = run_prediction(args.model_type, args.input)
        print("Inference completed successfully.")
        print(f"Prediction CSV: {result['csv']}")
        print(f"Plot: {result['plot']}")
        print(f"Log: {result['log']}")
    except Exception as err:
        print(str(err))
        sys.exit(1)
