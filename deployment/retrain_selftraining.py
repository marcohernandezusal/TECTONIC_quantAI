import os
import sys
import logging
import joblib
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import Ridge

from pipeline.models import SelfTrainingRegressor
from pipeline.data_loader import read_input_file, check_columns
from pipeline.evaluation import plot_predictions_array
from pipeline.utils import setup_logger, get_latest_model

SCALER_PATH = 'scaler/scaler.pkl'
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join("models", "selftraining")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_selftraining(input_file, st_logger=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log, log_path = setup_logger("retrain_selftraining", timestamp)
    log.info("=== Starting self-training retraining (continuous mode) ===")

    try:
        scaler = joblib.load(SCALER_PATH)
        df = read_input_file(input_file)
        expected_features = scaler.feature_names_in_
        check_columns(df, expected_features)

        X_raw = df[expected_features].copy()
        X_scaled = scaler.transform(X_raw)

        try:
            model_path = get_latest_model("selftraining")
            str_model = joblib.load(model_path)
            log.info(f"Loaded existing self-training model from: {model_path}")
        except FileNotFoundError:
            log.warning("No previous self-training model found. Starting fresh.")
            str_model = SelfTrainingRegressor(base_estimator=Ridge(), max_iter=10, verbose=True)

        y_initial_pred = str_model.predict(X_scaled)
        threshold = np.percentile(np.abs(y_initial_pred - np.mean(y_initial_pred)), 50)
        confident_idx = np.where(np.abs(y_initial_pred - np.mean(y_initial_pred)) <= threshold)[0]

        X_confident = X_scaled[confident_idx]
        y_confident = y_initial_pred[confident_idx]
        log.info(f"Selected {len(confident_idx)} confident pseudo-labels (threshold={threshold:.4f})")
        if st_logger:
            st_logger(f"🔍 Selected {len(confident_idx)} confident samples for retraining")

        str_model.fit(X_confident, y_confident, X_scaled)

        model_file = f"selftrained_model_{timestamp}.pkl"
        model_path = os.path.join(MODEL_DIR, model_file)
        joblib.dump(str_model, model_path)

        pointer_file = os.path.join("models", "latest_selftraining_model.txt")
        with open(pointer_file, "w") as f:
            f.write(model_path)

        metadata = {
            "model_type": "selftraining",
            "timestamp": timestamp,
            "features": list(expected_features),
            "input_source": os.path.basename(input_file),
            "scaler_used": SCALER_PATH
        }
        with open(model_path.replace(".pkl", ".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        y_pred = str_model.predict(X_scaled)
        df_out = df.copy()
        df_out["Predicted_Corrosion"] = y_pred

        csv_out = os.path.join(OUTPUT_DIR, f"predictions_selftrained_{timestamp}.csv")
        plot_file = os.path.join(OUTPUT_DIR, f"prediction_plot_selftrained_{timestamp}.png")
        df_out.to_csv(csv_out, index=False)
        plot_predictions_array(y_pred, plot_file)

        # Training progression plot
        train_plot_path = None
        if hasattr(str_model, "history_") and isinstance(str_model.history_, list) and str_model.history_:
            iter_nums = list(range(1, len(str_model.history_) + 1))
            pseudo_counts = [step.get("n_pseudo", 0) for step in str_model.history_]
            plt.figure(figsize=(6, 4), dpi=300)
            plt.plot(iter_nums, pseudo_counts, marker='o')
            plt.title("Pseudo-Labeling Progression")
            plt.xlabel("Iteration")
            plt.ylabel("# Pseudo-labeled samples")
            plt.grid(True)
            plt.tight_layout()
            train_plot_path = os.path.join(OUTPUT_DIR, f"training_plot_selftrained_{timestamp}.png")
            plt.savefig(train_plot_path)
            plt.close()
            log.info(f"Training progression plot saved to: {train_plot_path}")
            # if st_logger:
            #     st_logger("Training plot generated.")

        log.info(f"Retraining complete. Model saved: {model_path}")
        log.info(f"Prediction CSV: {csv_out}")
        log.info(f"Plot: {plot_file}")
        log.info(f"Log: {log_path}")

        if st_logger:
            st_logger("✅ Retraining complete.")
        #     st_logger(f"📁 Output CSV: {csv_out}")
        #     st_logger(f"🖼️ Prediction plot: {plot_file}")

        return {
            "csv": csv_out,
            "plot": plot_file,
            "log": log_path,
            "train_plot": train_plot_path
        }

    except Exception as e:
        log.exception(f"Retraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain self-training model with new data.")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    run_selftraining(args.input)