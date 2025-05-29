import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from pipeline.data_loader import load_and_split_data, scale_features, get_labeled_unlabeled_split
from pipeline.evaluation import evaluate_model, plot_predictions, plot_test_performance, plot_cv_scores
from pipeline.models import get_models_and_params, SelfTrainingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

# Setup logging
os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
logging.basicConfig(
    filename='logs/self_training_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def plot_confidence_histogram(confidences, model_name, iter_num):
    plt.figure()
    plt.hist(confidences, bins=50, alpha=0.7)
    plt.title(f"Confidence Histogram - {model_name} Iter {iter_num}")
    plt.xlabel("Prediction Std. Dev.")
    plt.ylabel("Count")
    plt.savefig(f"figures/confidence_hist_{model_name.replace(' ', '_')}_iter{iter_num}.png")
    plt.close()

def self_training_cv_pipeline(X_lab, y_lab, X_unlab, X_val, y_val, kf, base_models):
    results = {}
    best_model = None
    best_model_name = ""
    best_score = -np.inf

    for name, model_info in base_models.items():
        if len(model_info['params']) == 0:
            base_model = model_info['model']
        else:
            base_model = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=kf,
                scoring='r2',
                n_jobs=-1
            )

        logging.info(f"Running self-training for: {name}")
        str_model = SelfTrainingRegressor(base_estimator=base_model, verbose=True, threshold=0.1)
        str_model.fit(X_lab, y_lab, X_unlab, X_val, y_val, patience=2, log_conf_hist=True, model_name=name)

        scores = []
        for train_idx, test_idx in kf.split(X_val):
            X_ktrain, X_ktest = X_val[train_idx], X_val[test_idx]
            y_ktrain, y_ktest = y_val[train_idx], y_val[test_idx]
            try:
                str_model.fit(X_lab, y_lab, X_unlab, X_ktrain, y_ktrain)
                y_pred = str_model.predict(X_ktest)
                r2 = r2_score(y_ktest, y_pred)
                if np.isfinite(r2):
                    scores.append(r2)
            except Exception as e:
                logging.warning(f"{name} failed on CV fold: {e}")

        if len(scores) == 0:
            logging.warning(f"Skipping {name}, no valid CV scores.")
            continue

        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)
        results[name] = {'CV R2 Mean': mean_r2, 'CV R2 Std': std_r2}

        logging.info(f"{name} - Self-training CV R2: {mean_r2:.4f} ± {std_r2:.4f}")

        if mean_r2 > best_score:
            best_score = mean_r2
            best_model = str_model
            best_model_name = name

    return best_model, best_model_name, results

def main():
    logging.info("Starting self-training regression pipeline.")

    X_trainval, X_test, y_trainval, y_test = load_and_split_data("Dataset_Corrosion.csv", test_size=0.2)
    X_lab_raw, y_lab, X_unlab_raw = get_labeled_unlabeled_split(X_trainval, y_trainval, labeled_fraction=0.1)

    val_split = int(len(X_lab_raw) * 0.2)
    X_val_raw = X_lab_raw[:val_split]
    y_val = y_lab[:val_split]
    X_lab_raw = X_lab_raw[val_split:]
    y_lab = y_lab[val_split:]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_trainval)

    X_lab = scaler.transform(X_lab_raw)
    X_unlab = scaler.transform(X_unlab_raw)
    X_val = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)

    X_lab = np.array(X_lab)
    y_lab = np.array(y_lab)
    X_unlab = np.array(X_unlab)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    base_models = get_models_and_params()

    best_model, best_model_name, results = self_training_cv_pipeline(
        X_lab, y_lab, X_unlab, X_val, y_val, kf, base_models
    )

    plot_cv_scores(results, output_path="figures/self_training_cv_r2_scores.png")
    
    # Save validation metrics
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    val_metrics_df = pd.DataFrame.from_dict(results, orient='index')
    val_metrics_df.to_csv("results/self_training_cv_results.csv")
    logging.info("Validation metrics saved to results/self_training_cv_results.csv")

    best_model.fit(X_lab, y_lab, X_unlab, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test)

    logging.info("Test set performance of best model:")
    for metric, value in test_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # Save test set metrics
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("results/self_training_test_metrics.csv", index=False)
    logging.info("Test metrics saved to results/self_training_test_metrics.csv")

    plot_predictions(best_model, X_test_scaled, y_test, title="Self-Training Predictions vs Actual", filename="figures/self_training_predictions.png")
    plot_test_performance(test_metrics, output_path="figures/self_training_test_performance.png")

    joblib.dump(best_model, 'best_self_training_model.pkl')
    logging.info(f"Best self-training model saved to best_self_training_model.pkl")

if __name__ == '__main__':
    main()