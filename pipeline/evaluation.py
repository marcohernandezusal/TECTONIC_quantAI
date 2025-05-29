from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')

def debug_data(filepath):
    df = pd.read_csv(filepath, index_col=False, header=0)
    logging.info("\nDataFrame Info:\n" + str(df.info()))
    logging.info("\nDataFrame Describe:\n" + str(df.describe(include='all')))
    logging.info("\nDataFrame Head:\n" + str(df.head()))

    # Plot correlation heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("figures/correlation_matrix.png")
    plt.close()
    logging.info("Correlation matrix saved to figures/correlation_matrix.png")

    # Histogram of target
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Corrosion'], bins=30, kde=True)
    plt.title("Distribution of Corrosion")
    plt.tight_layout()
    plt.savefig("figures/corrosion_distribution.png")
    plt.close()
    logging.info("Corrosion target distribution saved to figures/corrosion_distribution.png")

    return df
    

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'R2 Score': r2_score(y_test, preds),
        'MSE': mean_squared_error(y_test, preds),
        'MAE': mean_absolute_error(y_test, preds)
    }

def plot_predictions(model, X, y_true, title="Predicted vs Actual", filename=None):
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        
def plot_feature_importance(importances, feature_names, model_name, output_path):
    if importances is None:
        logging.warning(f"No importances available for model: {model_name}")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5), dpi=300)

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = np.array(importances)[sorted_idx]

    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")

    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"{model_name} Feature Importances", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Feature importances plot saved: {output_path}")
    
def plot_cv_scores(results_dict, output_path="figures/model_cv_r2_scores.png"):
    """Plots model CV R² mean scores with standard deviation error bars."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)

    # Prepare DataFrame
    plot_df = pd.DataFrame([
        {'Model': model, 'Mean R2': data['CV R2 Mean'], 'Std R2': data['CV R2 Std']}
        for model, data in results_dict.items()
    ])
    plot_df.sort_values('Mean R2', inplace=True)

    # Plot with matplotlib
    plt.barh(
        plot_df['Model'],
        plot_df['Mean R2'],
        xerr=plot_df['Std R2'],
        color=sns.color_palette("mako", len(plot_df)),
        edgecolor='black'
    )

    plt.xlabel("R² Score (Mean ± Std)", fontsize=12)
    plt.xlim(0, 1)
    plt.title("Cross-Validation R² Scores per Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(f"CV R² score plot saved: {output_path}")


def plot_test_performance(metrics_dict, output_path="figures/best_model_test_performance.png"):
    """Plots bar chart of test set metrics for best model."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4), dpi=300)

    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())

    sns.barplot(x=metric_names, y=metric_values, palette="crest")
    plt.ylabel("Score", fontsize=12)
    plt.title("Best Model - Test Set Performance", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Best model test performance plot saved: {output_path}")
    
