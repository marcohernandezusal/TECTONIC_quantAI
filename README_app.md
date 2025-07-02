# 🧪 Corrosion Prediction in Underwater Cultural Heritage Environments

This repository hosts the **quantAI micro-service** for modelling and predicting corrosion in underwater-cultural-heritage (UCH) environments.  
The codebase now follows a clear **front ↔ back separation**:

```
┌─────────────┐  HTTP / JSON  ┌───────────────────────┐
│ Streamlit   │ — — — — — —▶ │ FastAPI “quantAI API” │
│ multipage   │               │  (uvicorn)            │
└─────────────┘               └────────────────────────┘
```

* **Streamlit** → user interface only (file upload, buttons, plots).  
* **FastAPI**   → runs the ML pipelines and stores metadata in SQLite  
  (or PostgreSQL via `DATABASE_URL`).

---

## 🗂 Repository structure

```text
.
├── Home.py                     # Streamlit landing page (renders this README)
├── pages/
│   └── 1_Quant_models.py       # UI that calls the API and visualises results
├── utils_front.py              # Shared helpers for the Streamlit pages
├── fastapi_main.py             # FastAPI backend (models, sessions, DB, /static)
│
├── deployment/                 # Thin wrappers reused by the API
│   ├── predict_model.py
│   └── retrain_selftraining.py
├── pipeline/                   # Core data / modelling code (unchanged)
├── training/                   # Training notebooks, figures & scripts
├── outputs/ , models/ , scaler/ , Dataset_Corrosion.csv
└── requirements.txt , README_app.md
```

> **Legacy file** `front_sim.py` has been replaced by the multipage UI and can be removed.

---

## 📊 Problem overview

We predict corrosion levels from environmental factors — **Temperature, Salinity, Pressure** — to guide conservation strategies for UCH artefacts.

---

## 📁 Dataset description

The dataset `Dataset_Corrosion.csv` was compiled by the BISITE research group (Carolina Villoria Torres & Juan Manuel Núñez Velasco):

* Readings sourced from **SeaDataNet CDI** & **ODATIS-Coriolis**.  
* Locations closest to the **TECTONIC pilot sites**.  
* **Corrosion** inferred via **Kriging interpolation** (Cressie & Johannesson, 2008) using data from **Wang et al., 2021**.

<details>
<summary><strong>References</strong></summary>

* Wang, Z., Sobey, A. J., & Wang, Y. (2021). *Materials & Design*, 208, 109910.  
* Cressie, N., & Johannesson, G. (2008). *JRSS B*, 70 (1), 209-226.
</details>

---

### 🔬 Target distribution

![Corrosion distribution](/static/corrosion_distribution.png)

---

## 🧠 Supervised learning pipeline

```bash
python training/supervised_models.py
```

* 5-fold CV + grid-search, best model selected by R², evaluated on a hold-out set.

| Result | Figure |
|--------|--------|
| **CV scores** | ![CV scores](/static/model_cv_r2_scores.png) |
| **Best model test performance** | ![Best model test perf](/static/best_model_test_performance.png) |
| **Predictions vs actual** | ![Predictions vs actual](/static/pred_vs_actual.png) |

---

## 🤖 Semi-supervised learning (self-training)

```bash
python training/self_training_models.py
```

* Starts with 10 % labelled data, iteratively pseudo-labels and retrains.

| Result | Figure |
|--------|--------|
| **CV scores** | ![Self-training CV](/static/self_training_cv_r2_scores.png) |
| **Predictions vs actual** | ![Self-training predictions](/static/self_training_predictions.png) |
| **Final test performance** | ![Self-training test perf](/static/self_training_test_performance.png) |

---

## 🧬 Feature correlations

![Correlation matrix](/static/correlation_matrix.png)

---

## 🖧 quantAI API (FastAPI)

| Verb | Route | Purpose |
|------|-------|---------|
| `POST` | `/sessions` | Create or reuse a session (returns UUID). |
| `POST` | `/predict` | Run supervised prediction on uploaded file. |
| `POST` | `/selftraining` | Run self-training; optional `retrain=true`. |
| `GET`  | `/predictions/{id}/csv` | Download CSV results for that prediction. |
| `GET`  | `/predictions/{id}/plot` | Download PNG plot. |
| `GET`  | `/models`, `/models/{id}` | Register & list models (future). |

The default DB is SQLite (`sqlite:///./tectonic_ai.db`).  
Override with the `DATABASE_URL` environment variable if you prefer Postgres.

---

## © License

MIT License — research & educational use.
