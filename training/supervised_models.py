import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add previous directory ('..') to path to import local modules
import sys
# --- Set up base directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # Folder of this script
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))        # Parent folder (for pipeline/)

# --- Add parent directory to sys.path so we can import 'pipeline' ---
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from pipeline.data_loader import load_and_split_data, scale_features
from pipeline.models import get_models_and_params
from sklearn.model_selection import GridSearchCV
from pipeline.evaluation import *
from sklearn.model_selection import cross_val_score, KFold
import joblib

# Setup logging
os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def main():
    logging.info("Starting corrosion regression pipeline.")

    # Load and preprocess data
    X_trainval, X_test, y_trainval, y_test = load_and_split_data("../Dataset_Corrosion.csv", test_size=0.2)
    scaler, X_trainval_scaled, X_test_scaled = scale_features(X_trainval, X_test, return_scaler=True)
    scaler_path = "../scaler/scaler_supervised.pkl"
    os.makedirs("../scaler", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info("Scaler saved to scaler/scaler_supervised.pkl")
    # scale target variable
    y_trainval_scaled = (y_trainval - y_trainval.min()) / (y_trainval.max() - y_trainval.min())
    y_test_scaled = (y_test - y_test.min()) / (y_test.max() - y_test.min())
    
    logging.info("Data loaded and scaled.")

    logging.info(f"y_trainval stats: min={y_trainval.min()}, max={y_trainval.max()}, mean={y_trainval.mean()}")

    # 5-fold cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_and_params = get_models_and_params()
    models = {
        name: GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='r2', n_jobs=-1)
        for name, model_info in models_and_params.items()
    }

    best_model = None
    best_model_name = ""
    best_score = -np.inf
    results = {}

    logging.info("Performing 5-fold cross-validation:")
    for name, model in models.items():
        logging.info(f"Starting hyperparameter tuning for: {name}")
        
        model.fit(X_trainval_scaled, y_trainval)

        if hasattr(model, 'best_params_'):
            logging.info(f"{name} best parameters: {model.best_params_}")
        if hasattr(model, 'best_score_'):
            logging.info(f"{name} best CV score (internal): {model.best_score_:.4f}")
            
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importances = getattr(model, 'feature_importances_', None)
            if importances is None:
                importances = np.abs(getattr(model, 'coef_', None))
            plot_feature_importance(
                importances,
                X_trainval.columns if isinstance(X_trainval, pd.DataFrame) else [f'Feature {i}' for i in range(X_trainval.shape[1])],
                name,
                f"figures/feature_importance_{name.replace(' ', '_')}.png"
            )
        
        scores = cross_val_score(model, X_trainval_scaled, y_trainval, cv=kf, scoring='r2')
        mean_score = scores.mean()
        std_score = scores.std()
        results[name] = {'CV R2 Mean': mean_score, 'CV R2 Std': std_score}

        logging.info(f"{name} - R2 mean score: {mean_score:.4f}, std: {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name
            
    plot_cv_scores(results)
    # Save validation metrics
    os.makedirs("results", exist_ok=True)
    val_metrics_df = pd.DataFrame.from_dict(results, orient='index')
    val_metrics_df.to_csv("results/supervised_cv_results.csv")
    logging.info("Validation metrics saved to results/supervised_cv_results.csv")

    # Train best model on full trainval set
    best_model.fit(X_trainval_scaled, y_trainval_scaled)
    joblib.dump(best_model, 'best_model.pkl')
    logging.info(f"Best model selected: {best_model_name}, saved to best_model.pkl")

    # Evaluate on trainval set for debugging
    trainval_metrics = evaluate_model(best_model, X_trainval_scaled, y_trainval_scaled)
    logging.info("Trainval set performance:")
    for metric, value in trainval_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # Evaluate on test set
    test_metrics = evaluate_model(best_model, X_test_scaled, y_test_scaled)
    results['Best Model Test Set'] = test_metrics
    logging.info("Test set performance:")
    for metric, value in test_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
        
    # Save test set metrics
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("results/supervised_test_metrics.csv", index=False)
    logging.info("Test metrics saved to results/supervised_test_metrics.csv")

    # Plot predictions vs actual and save
    plot_predictions(best_model, X_test_scaled, y_test_scaled, title="Best Model Predictions vs Actual", filename="figures/pred_vs_actual.png")
    plot_test_performance(test_metrics)
    logging.info("Plot saved to figures/pred_vs_actual.png")

    logging.info("Pipeline execution complete.")

if __name__ == '__main__':
    main()