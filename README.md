# 🧪 Corrosion Prediction in Underwater Cultural Heritage Environments

This repository hosts the **quantAI micro-service** for modelling and predicting corrosion in underwater cultural-heritage (UCH) environments.  
It now follows a **front ↔ back separation**:

```
┌─────────────┐ HTTP/JSON  ┌───────────────────────┐
│ Streamlit   │──────────▶│ FastAPI “quantAI API” │
│ multipage   │           │  (uvicorn)            │
└─────────────┘           └────────────────────────┘
```

* **Streamlit** – user interface only (file upload, buttons, plots).  
* **FastAPI**   – runs the ML pipelines and stores metadata in SQLite (or Postgres).

---

## 🗂 Repository Structure

```text
.
├── Home.py                     # Streamlit landing page (renders this README)
├── pages/
│   └── 1_Quant_models.py       # UI to call the API and visualise results
├── utils_front.py              # Shared helpers for the Streamlit pages
├── fastapi_main.py             # FastAPI backend (models, sessions, DB)
│
├── deployment/                 # Thin wrappers reused by the API
│   ├── predict_model.py
│   └── retrain_selftraining.py
├── pipeline/                   # Core data / modelling code (unchanged)
├── training/                   # Training notebooks, figures & scripts
├── outputs/ , models/ , scaler/ , Dataset_Corrosion.csv
└── requirements.txt , README.md
```

> **Legacy file** `front_sim.py` has been superseded by the multipage UI and can be deleted.

---

## 📊 Problem Overview

We predict corrosion levels from environmental factors – **Temperature, Salinity, Pressure** – to support conservation strategies for UCH artefacts.

---

## 📁 Dataset Description

The dataset `Dataset_Corrosion.csv` was compiled by the BISITE research group (Carolina Villoria Torres & Juan Manuel Núñez Velasco):

* Readings sourced from **SeaDataNet CDI** & **ODATIS-Coriolis**.
* Locations closest to **TECTONIC pilot sites**.
* **Corrosion** inferred via **Kriging interpolation** (Cressie & Johannesson 2008) using data from **Wang et al. 2021**.

<details>
<summary><strong>References</strong></summary>

* Wang, Z., Sobey, A. J., & Wang, Y. (2021). *Materials & Design*, 208, 109910.  
* Cressie, N., & Johannesson, G. (2008). *JRSS B*, 70 (1), 209-226.
</details>

---

### 🔬 Target Distribution

![Corrosion Distribution](training/figures/corrosion_distribution.png)

---

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

## 🖧 quantAI API (FastAPI)

| Verb | Route | Purpose |
|------|-------|---------|
| **POST** | `/sessions` | Create or reuse a session (returns UUID). |
| **POST** | `/predict` | Run supervised prediction on uploaded file. |
| **POST** | `/selftraining` | Run self-training; optional `retrain=true`. |
| **GET**  | `/predictions/{id}/csv` | Download CSV results for that prediction. |
| **GET**  | `/predictions/{id}/plot` | Download PNG plot. |
| **GET**  | `/models`, `/models/{id}` | Register & list models (future). |

Default DB is SQLite (`sqlite:///./tectonic_ai.db`); override with `DATABASE_URL`.

Start the backend:

```bash
uvicorn fastapi_main:app --reload
```

---

## 🌐 Interactive Web Interface (Streamlit multipage)

```bash
# 1) Launch backend (above)
# 2) Optional – point front-end to a remote backend
export API_URL="http://localhost:8000"        # Windows CMD: set API_URL=...
# 3) Start UI
streamlit run Home.py
```

### Front-end features
* 📤 Upload `.csv` / `.xlsx`
* 🧠 Choose **supervised** or **selftraining**
* 🔁 Optional retraining flag for self-training
* 📈 Visualise predictions & pseudo-labelling
* 📥 Download predictions (CSV) & logs

![Frontend Screenshot](training/figures/streamlit_view.png)

---

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key libraries: `fastapi`, `uvicorn`, `httpx`, `sqlalchemy`, `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`.

---

## © License
MIT License – research & educational use.
