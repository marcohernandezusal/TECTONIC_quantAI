import os
import logging

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone



class PTMLPRegressor(BaseEstimator, RegressorMixin):
    """
    A simple feed-forward MLP in PyTorch with optional L1 regularization.
    """
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        lr=1e-3,
        batch_size=32,
        max_epochs=100,
        l1_alpha=0.0,
        random_state=42,
        device=None
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.l1_alpha = l1_alpha
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_network(self, n_features):
        layers = []
        in_size = n_features
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh}[self.activation]
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        # reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.model_ = self._build_network(self.n_features_in_)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model_(xb)
                loss = criterion(preds, yb)
                if self.l1_alpha > 0:
                    l1_penalty = sum(p.abs().sum() for p in self.model_.parameters())
                    loss = loss + self.l1_alpha * l1_penalty
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            # you could add early stopping here if you like
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        tensor = torch.from_numpy(X).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(tensor).cpu().numpy().ravel()
        return out

def get_models_and_params():

    models_and_params = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 20, None]
            }
        },
        'Gradient Boosting': {
            'model': HistGradientBoostingRegressor(random_state=42),
            'params': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'max_iter': [100, 200],
                'early_stopping': [True]
            }
        },
        'SVR': {
            'model': make_pipeline(StandardScaler(), SVR()),
            'params': {
                'svr__C': [0.1, 1, 10],
                'svr__epsilon': [0.01, 0.1, 0.2]
            }
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, None]
            }
        },
        'MLP': {
            'model': make_pipeline(
                StandardScaler(),
                MLPRegressor(max_iter=1000, random_state=42)
            ),
            'params': {
                'mlpregressor__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'mlpregressor__activation': ['relu', 'tanh'],
                'mlpregressor__alpha': [0.0001, 0.001],
                'mlpregressor__learning_rate_init': [0.001, 0.01]
            }
        },
        # 'MLP (pytorch)': {
        #     'model': make_pipeline(StandardScaler(),
        #                            PTMLPRegressor(random_state=42)),
        #     'params': {
        #         'ptmlpregressor__hidden_layer_sizes': [(10,), (50,), (100,), (100, 50)],
        #         'ptmlpregressor__activation': ['relu', 'tanh'],
        #         'ptmlpregressor__lr': [1e-2, 1e-3, 1e-4],
        #         'ptmlpregressor__batch_size': [32, 64],
        #         'ptmlpregressor__max_epochs': [50, 100],
        #         'ptmlpregressor__l1_alpha': [0.0, 1e-4, 1e-3]
        #     }
        # },
        'Voting Regressor': {
            'model': VotingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                    ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
                    ('ridge', Ridge(alpha=1.0))
                ]
            ),
            'params': {}
        },
        'Stacking Regressor': {
            'model': StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                    ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
                ],
                final_estimator=Ridge(alpha=1.0)
            ),
            'params': {}
        }
    }
    return models_and_params

class SelfTrainingRegressor(BaseEstimator, RegressorMixin):
    """
    Self-training wrapper for any regressor.  
    Iteratively pseudolabels unlabeled data with high-confidence predictions.
    Logs per-iteration counts and confidence statistics.
    """
    def __init__(
        self,
        base_estimator,
        threshold=0.3,
        max_iter=10,
        verbose=True
    ):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose
        self.history_ = []  # will store dicts of iteration stats
        self.logger = logging.getLogger(__name__)
        
    def plot_confidence_histogram(self, confidences, model_name, iter_num):
            os.makedirs("figures/confidence", exist_ok=True)
            plt.figure()
            plt.hist(confidences, bins=50, alpha=0.7)
            plt.title(f"Confidence Histogram - {model_name} Iter {iter_num}")
            plt.xlabel("Prediction Std. Dev.")
            plt.ylabel("Count")
            plt.savefig(f"figures/confidence/confidence_hist_{model_name.replace(' ', '_')}_iter{iter_num}.png")
            plt.close()
            
    def fit(self, X_labeled, y_labeled, X_unlabeled, X_val=None, y_val=None, patience=1, log_conf_hist=False, model_name=None):
        X_lab = np.array(X_labeled)
        y_lab = np.array(y_labeled)
        X_unlab = np.array(X_unlabeled)

        self.history_.clear()
        prev_val_r2 = None
        patience_counter = 0

        for it in range(1, self.max_iter + 1):
            if len(X_unlab) == 0:
                self.logger.info(f"No more unlabeled data at iteration {it}. Stopping.")
                break

            self.base_estimator.fit(X_lab, y_lab)

            # Validation check
            if X_val is not None and y_val is not None:
                val_pred = self.base_estimator.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)

                if prev_val_r2 is not None and val_r2 < prev_val_r2 - 0.02:
                    patience_counter += 1
                    self.logger.info(f"Validation R² dropped: {prev_val_r2:.3f} → {val_r2:.3f} (patience {patience_counter}/{patience})")
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at iteration {it} due to validation R² drop.")
                        break
                else:
                    patience_counter = 0

                prev_val_r2 = val_r2

            # Predict on unlabeled data
            preds = self.base_estimator.predict(X_unlab)

            # Estimate confidence
            if hasattr(self.base_estimator, 'estimators_'):
                all_tree_preds = np.stack([e.predict(X_unlab) for e in self.base_estimator.estimators_])
                conf = np.std(all_tree_preds, axis=0)
            else:
                conf = np.abs(preds - np.mean(y_lab))  # fallback

            # Optional: log histogram
            if log_conf_hist and model_name:
                self.plot_confidence_histogram(conf, model_name, it)


            thr = np.quantile(conf, 0.25)
            mask = conf <= thr
            new_count = mask.sum()

            self.logger.info(f"Model: {model_name},Iter {it}: new pseudo-labeled points = {new_count}, conf threshold = {thr:.4f}")
            if new_count == 0:
                break

            X_new = X_unlab[mask]
            y_new = preds[mask]
            X_lab = np.vstack([X_lab, X_new])
            y_lab = np.concatenate([y_lab, y_new])
            X_unlab = X_unlab[~mask]

            self.logger.info(f"Unlabeled pool size after iter {it}: {len(X_unlab)}")

        self.base_estimator.fit(X_lab, y_lab)
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)