# ==============================================================
# Train MANY models, keep ONLY the best (by Test_RMSE)
#   - Robust to unknown cities
#   - Live-updatable KNNLocalPrice feature
#   - Non-negative predictions via log1p-target training
#   - Saves ONLY:
#       - artifacts/best_model_<Name>_<timestamp>.joblib
#       - artifacts/run_meta.json (includes best_model_weights)
# ==============================================================

import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    GammaRegressor,  # positive continuous targets
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------- Paths -----------------------
ARTIFACT_DIR = os.path.join("artifacts")

# ----------------------- Columns ---------------------
FORBIDDEN_COLS = {"city", "postal_code"}

# Tiny epsilon you can use in your API to enforce strictly > 0
TINY_POS = np.finfo(float).tiny


def drop_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[c for c in FORBIDDEN_COLS if c in df.columns], errors="ignore"
    )


# ----------------- Custom transformers ----------------
class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Cross-validated-style target encoder with smoothing.
    NOTE: Not used by default below, but kept for convenience.
    """

    def __init__(
        self,
        cols: List[str],
        n_splits: int = 5,
        smoothing: float = 50.0,
        random_state: int = 42,
    ):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: Optional[float] = None
        self.maps_: Dict[str, Dict[str, float]] = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y)
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            stats = (
                pd.DataFrame({col: X[col], "y": y})
                .groupby(col)["y"]
                .agg(["mean", "count"])
            )
            w = 1.0 / (1.0 + np.exp(-(stats["count"] - self.smoothing)))  # [0,1]
            te = self.global_mean_ * (1.0 - w) + stats["mean"] * w
            self.maps_[col] = te.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out_cols = []
        for col in self.cols:
            mapped = X[col].map(self.maps_[col]).fillna(self.global_mean_)
            out_cols.append(mapped.rename(col + "_te").to_frame())
        return pd.concat(out_cols, axis=1).values

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_te" for c in self.cols], dtype=object)


class KNNLocalPrice(BaseEstimator, TransformerMixin):
    """
    Live-updatable KNN local price feature:
      - fit(X_ll, y): store training (lat, lon) and target, build index
      - transform(X_ll): return mean price of K nearest neighbors
      - update(X_ll_new, y_new): append new labeled points and rebuild index
    """

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self._nn: Optional[NearestNeighbors] = None
        self._xy: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                "KNNLocalPrice expects exactly two columns: [latitude, longitude]"
            )
        y = np.asarray(y, dtype=float).ravel()
        self._xy = X.copy()
        self._y = y.copy()
        self._rebuild()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self._nn is None or self._y is None:
            raise RuntimeError("KNNLocalPrice must be fitted before transform.")
        k = min(self.n_neighbors, len(self._xy))
        _, indices = self._nn.kneighbors(X, n_neighbors=k, return_distance=True)
        means = self._y[indices].mean(axis=1)
        return means.reshape(-1, 1)

    def update(self, X_new, y_new):
        X_new = np.asarray(X_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float).ravel()
        if X_new.ndim != 2 or X_new.shape[1] != 2:
            raise ValueError(
                "update expects X_new with shape (n, 2) [latitude, longitude]"
            )
        if len(X_new) != len(y_new):
            raise ValueError("Lengths of X_new and y_new must match")
        if self._xy is None:
            self._xy = X_new.copy()
            self._y = y_new.copy()
        else:
            self._xy = np.vstack([self._xy, X_new])
            self._y = np.concatenate([self._y, y_new])
        self._rebuild()

    def _rebuild(self):
        n = max(1, min(self.n_neighbors, len(self._xy)))
        self._nn = NearestNeighbors(n_neighbors=n, metric="euclidean")
        self._nn.fit(self._xy)

    def get_feature_names_out(self, input_features=None):
        return np.array(["local_price_knn"], dtype=object)


# ----------------------- Helpers -----------------------
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Robust preprocessor (NO 'nearest_hesso_name'; keep proxim_hesso_km):
      - Numerics (scaled): surface_m2, num_rooms, floor, dist_public_transport_km, proxim_hesso_km, latitude, longitude
      - OneHot: type
      - Booleans: passthrough
      - Geo KNN feature from [latitude, longitude]
    """
    default_num = [
        "surface_m2",
        "num_rooms",
        "floor",
        "dist_public_transport_km",
        "proxim_hesso_km",
        "latitude",
        "longitude",
    ]
    default_cat_ohe = ["type"]
    default_bool = ["is_furnished", "wifi_incl", "charges_incl", "car_park"]

    num_features = [c for c in default_num if c in X.columns]
    cat_ohe_features = [c for c in default_cat_ohe if c in X.columns]
    bool_features = [c for c in default_bool if c in X.columns]

    ohe = _make_ohe()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", ohe, cat_ohe_features),
            ("bool", "passthrough", bool_features),
            ("geo_knn", KNNLocalPrice(n_neighbors=10), ["latitude", "longitude"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def make_pos_ttr(estimator) -> TransformedTargetRegressor:
    """
    Train on log1p(y) and invert with expm1 to guarantee non-negative outputs.
    Uses NumPy built-ins to remain pickle-safe under Uvicorn/Gunicorn.
    """
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _unwrap_final_estimator(
    final_estimator: BaseEstimator,
) -> Tuple[BaseEstimator, str]:
    """
    If the final estimator is a TransformedTargetRegressor, return its fitted regressor_.
    """
    if isinstance(final_estimator, TransformedTargetRegressor):
        try:
            return (
                final_estimator.regressor_,
                "TransformedTargetRegressor(log1p->expm1)",
            )
        except Exception:
            return (
                final_estimator.regressor,
                "TransformedTargetRegressor(log1p->expm1, unfitted unwrap)",
            )
    return final_estimator, "none"


def _get_feature_names(preprocessor: ColumnTransformer) -> Optional[List[str]]:
    """Try to fetch feature names after preprocessing to align with coef_/importances."""
    try:
        names = preprocessor.get_feature_names_out()
        return list(map(str, names))
    except Exception:
        return None


def _extract_weights(pipe: Pipeline) -> Dict[str, Any]:
    """
    Extract interpretable "weights" from the best model.

    Returns dict with:
    - type: "coef", "feature_importances", or "unavailable"
    - wrapper: info about TTR wrapping, if any
    - intercept: float or None
    - weights: list of {feature, value} when feature names available, else raw list
    """
    info: Dict[str, Any] = {
        "type": "unavailable",
        "wrapper": None,
        "intercept": None,
        "weights": None,
    }

    try:
        pre: ColumnTransformer = pipe.named_steps["preprocess"]
        final = pipe.named_steps["regressor"]
    except Exception:
        return info

    est, wrapper = _unwrap_final_estimator(final)
    info["wrapper"] = wrapper
    feature_names = _get_feature_names(pre)

    # Linear-family with coef_
    if hasattr(est, "coef_"):
        coefs = np.asarray(getattr(est, "coef_"))
        if coefs.ndim == 1:
            coefs_list = coefs.tolist()
            if feature_names is not None and len(feature_names) == len(coefs_list):
                weights = [
                    {"feature": f, "value": float(w)}
                    for f, w in zip(feature_names, coefs_list)
                ]
            else:
                weights = coefs_list
        else:
            weights = coefs.tolist()
        info.update(
            {
                "type": "coef",
                "intercept": float(getattr(est, "intercept_", 0.0))
                if hasattr(est, "intercept_")
                else None,
                "weights": weights,
            }
        )
        return info

    # Tree/ensemble with feature_importances_
    if hasattr(est, "feature_importances_"):
        imps = np.asarray(getattr(est, "feature_importances_")).tolist()
        if feature_names is not None and len(feature_names) == len(imps):
            weights = [
                {"feature": f, "value": float(v)} for f, v in zip(feature_names, imps)
            ]
        else:
            weights = imps
        info.update({"type": "feature_importances", "weights": weights})
        return info

    # Others (SVR RBF, KNN, MLP) -> unavailable
    return info


# ------------------- Core: train & save best -------------------
def train_and_save_best(
    df: pd.DataFrame,
    target: str = "price_chf",
    out_dir: str = ARTIFACT_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains many regressors (log1p target -> expm1 inverse),
    evaluates on a held-out test split, saves ONLY the best
    pipeline (.joblib), and writes run_meta.json with full per-model test stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"📂 Artifacts dir: {out_dir}")
    print(f"🎯 Target: {target}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    # Optionally drop forbidden columns if present
    df = drop_forbidden(df)

    X, y = df.drop(columns=[target]), df[target].astype(float)

    # Sanity: warn if any y < 0 (invalid for log1p), and clip to 0
    if np.any(y < 0):
        n_neg = int(np.sum(y < 0))
        print(
            f"⚠️  Found {n_neg} negative target value(s). They will be clipped to 0 for log1p."
        )
        y = y.clip(lower=0.0)

    # Split BEFORE fitting any preprocessing to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Candidate base models (to be wrapped with log1p/expm1)
    base_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=100000),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=random_state
        ),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, random_state=random_state
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "SVR (RBF Kernel)": SVR(kernel="rbf", C=100, gamma=0.1),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=5000, random_state=random_state
        ),
    }

    # Wrap all general-purpose models with non-negative target transform
    models: Dict[str, BaseEstimator] = {
        name: make_pos_ttr(est) for name, est in base_models.items()
    }

    # Add GLM for positive continuous target (Gamma; mean is strictly positive)
    models.update(
        {"Gamma Regressor (log-link)": GammaRegressor(alpha=1.0, max_iter=2000)}
    )

    results = []  # collect stats for all models
    best_name, best_pipe, best_rmse = None, None, float("inf")

    for idx, (name, est) in enumerate(models.items(), start=1):
        print(f"\n🚀 [{idx}/{len(models)}] Training {name}...")
        # Build a fresh preprocessor per pipeline so it's fit ONLY on train
        pre = build_preprocessor(X)
        pipe = Pipeline([("preprocess", pre), ("regressor", est)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        # Numerical safety; if you *require* strictly > 0, enforce that in your API layer.
        preds = np.maximum(preds, 0.0)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        stats = {
            "model": name,
            "Test_RMSE": float(rmse),
            "Test_MAE": float(mae),
            "Test_R2": float(r2),
        }
        results.append(stats)
        print(f"   🔢 {stats}")

        if rmse < best_rmse:
            best_name, best_pipe, best_rmse = name, pipe, rmse
            print(f"   🥇 New best so far: {best_name} (RMSE={best_rmse:.4f})")

    if best_pipe is None:
        raise RuntimeError("No model was trained successfully.")

    # Save best pipeline
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = best_name.replace(" ", "_").replace("/", "_")
    model_path = os.path.join(out_dir, f"best_model_{safe_name}_{ts}.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n💾 Saved best model: {best_name} (RMSE={best_rmse:.4f}) → {model_path}")

    # Extract weights of the best model for the meta JSON
    best_weights = _extract_weights(best_pipe)

    # Save full run metadata with all per-model test stats
    meta = {
        "target": target,
        "random_state": random_state,
        "test_size": test_size,
        "created_at": ts,
        "best_model": best_name,
        "best_test_rmse": float(best_rmse),
        "model_path": model_path,
        "best_model_weights": best_weights,
        "all_results": results,  # full leaderboard-like stats
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata saved → {os.path.join(out_dir, 'run_meta.json')}")

    return best_pipe, meta


# ----------------------- CLI -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_models.py /path/to/data.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = drop_forbidden(df)
    # If your dataset has negatives, they are clipped to 0 before log1p.
    train_and_save_best(df, target="price_chf", out_dir=ARTIFACT_DIR)
