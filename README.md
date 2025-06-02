# 🧪 Corrosion Prediction in Underwater Cultural Heritage Environments

This repository contains machine learning pipelines for modeling and predicting corrosion in underwater cultural heritage (UCH) environments using both **supervised** and **semi-supervised (self-training)** regression techniques.

---

## 🗂 Repository Structure

.
├── front_sim.py                   # Streamlit web frontend
├── deployment/
│   ├── predict_model.py          # Inference logic for supervised & self-training models
│   └── retrain_selftraining.py   # Self-training retraining script with Streamlit logging
├── pipeline/
│   ├── data_loader.py            # Data preprocessing and validation
│   ├── evaluation.py             # Model evaluation and plotting
│   ├── models.py                 # Base and self-training model definitions
│   └── utils.py                  # Utility functions (logging, file management)
├── outputs/                      # Inference results and figures
├── models/                       # Stored models and pointer to latest
├── training/                     # Model training scripts and outputs
│   ├── self_training_models.py   # Self-training pipeline
│   ├── supervised_models.py      # Supervised learning pipeline
│   ├── EDA.ipynb                 # EDA notebook
│   ├── logs/                     # Execution logs
│   ├── figures/                  # Plots for EDA and model evaluation
│   └── results/                  # Cross-validation and test metrics
├── scaler/                       # Stored scalers
├── Dataset_Corrosion.csv         # Input dataset
└── README.md

---

## 🌐 Interactive Web Interface

Launch the frontend interface with Streamlit:
```bash
streamlit run front_sim.py
```

### Features:
- 📤 Upload `.csv` or `.xlsx` files with environmental data
- 🧠 Select between **supervised** and **selftraining** models
- 🔁 Optionally enable **retraining** when using selftraining mode
- 📈 Visualize:
  - Corrosion predictions
  - Pseudo-labeling progression during selftraining
- 📥 Download:
  - Predictions (CSV)
  - Logs (optional)

Example screenshot:

![Frontend Screenshot](training/figures/streamlit_view.png)

---

## 📊 Problem Overview

We aim to predict corrosion levels from environmental factors such as **Temperature**, **Salinity**, and **Pressure**. These predictions will help guide conservation strategies for UCH artifacts.

---

## 📁 Dataset Description

The dataset `Dataset_Corrosion.csv` was compiled by members of the BISITE research group (Carolina Villoria Torres and Juan Manuel Núñez Velasco). It contains:

- **Temperature**, **Salinity**, and **Pressure** readings collected from **SeaDataNet CDI** and **ODATIS-Coriolis** datasets.
- Data corresponds to the nearest available locations to the **TECTONIC pilot sites**.

Since no direct corrosion measurements exist, the **Corrosion** variable was inferred using **Kriging interpolation** (Cressie & Johannesson, 2008), based on data tables from **Wang et al. (2021)**.

### 📚 References:
- Wang, Z., Sobey, A. J., & Wang, Y. (2021). Corrosion prediction for bulk carrier via data fusion of survey and experimental measurements. *Materials & Design*, 208, 109910.
- Cressie, N., & Johannesson, G. (2008). Fixed rank kriging for very large spatial data sets. *Journal of the Royal Statistical Society: Series B*, 70(1), 209–226.

---

### 🔬 Target Distribution

![Corrosion Distribution](training/figures/corrosion_distribution.png)

---

## 🧠 Supervised Learning Pipeline

This script:
- Trains multiple regression models using 5-fold CV and grid search.
- Selects the best model based on R² score.
- Evaluates final performance on an unseen test set.

To run:
```bash
python supervised_models.py
```

### 📈 Cross-Validation Results

![CV Scores](training/figures/model_cv_r2_scores.png)

### ✅ Best Model Test Performance

![Supervised Test Perf](training/figures/best_model_test_performance.png)

### 🎯 Predictions vs Actual

![Supervised Predictions](training/figures/pred_vs_actual.png)

---

## 🤖 Semi-Supervised Learning (Self-Training)

This pipeline uses only 10% of labeled data and gradually augments training data using confident predictions from the model itself.

To run:
```bash
cd training
python main_self_training.py
```

### 🧪 CV Results Across Models

![Self-Training CV](training/figures/self_training_cv_r2_scores.png)

### 🎯 Predictions vs Actual

![Self-Training Predictions](training/figures/self_training_predictions.png)

### ✅ Final Test Performance

![Self-Training Test Perf](training/figures/self_training_test_performance.png)

---

## 🧬 Feature Correlations

Correlation matrix between input features and the corrosion target:

![Correlation Matrix](training/figures/correlation_matrix.png)

---

## 📋 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Main libraries:
- `streamlit`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `joblib`

---

## © License

MIT License. For research and educational use.

---

## 📧 Contact

For questions or collaboration inquiries, please reach out to the maintainers.
