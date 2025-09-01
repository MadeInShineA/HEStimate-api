# ==============================================================
# Train MANY models, keep ONLY the best (by Test_RMSE)
#   - Robust to unknown cities
#   - Live-updatable KNNLocalPrice feature
#   - Strictly positive predictions via log-target training (no clipping)
#   - Saves ONLY:
#       - artifacts/best_model_<Name>_<timestamp>.joblib
#       - artifacts/run_meta.json (now with *final* labeled weights)
# ==============================================================

import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------- Paths -----------------------
ARTIFACT_DIR = os.path.join("artifacts")

# ----------------------- Columns ---------------------
FORBIDDEN_COLS = {"city", "postal_code"}


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


def _to_jsonable(x: Any) -> Any:
    """Convert numpy/scalar types to JSON-friendly Python types."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _pair_labels(
    values: np.ndarray, names: List[str], key_name: str
) -> List[Dict[str, Any]]:
    """Return list of {feature, key_name} dicts for readability."""
    return [
        {"feature": str(n), key_name: _to_jsonable(v)} for n, v in zip(names, values)
    ]


def _linear_weights_in_original_space(
    pipe: Pipeline, coef_z: np.ndarray, intercept_z: float
) -> Tuple[List[str], np.ndarray, float]:
    """
    Map linear coefficients from standardized 'z' space back to original inputs.
    - coef_z / intercept_z are for the *transformed* features produced by the ColumnTransformer.
    - We undo StandardScaler for the numeric block; other blocks (one-hot, booleans, geo_knn) are already in "original" interpretation.
    Returns (feature_names, coef_original, intercept_original) all in LOG-PRICE space.
    """
    pre: ColumnTransformer = pipe.named_steps["preprocess"]
    feat_names = list(pre.get_feature_names_out())

    coef_z = np.asarray(coef_z).ravel().copy()
    intercept = float(intercept_z)

    # Figure out lengths of each block to know indices
    # We defined transformers in order: num, cat, bool, geo_knn
    # --- NUM ---
    num_cols = pre.transformers_[0][2]  # list of original numeric col names
    scaler: StandardScaler = pre.named_transformers_["num"]
    num_len = len(num_cols)
    # --- CAT ---
    cat_cols = pre.transformers_[1][2]
    if len(cat_cols) > 0:
        cat_ohe = pre.named_transformers_["cat"]
        cat_len = len(cat_ohe.get_feature_names_out(cat_cols))
    else:
        cat_len = 0
    # --- BOOL ---
    bool_cols = pre.transformers_[2][2]
    bool_len = len(bool_cols)
    # --- GEO_KNN ---
    geo_len = 1  # KNNLocalPrice returns a single feature

    # Adjust numeric block: z = (x - mean)/scale  =>  w*x = (w/scale)*x  and intercept -= w*mean/scale
    if num_len:
        scales = np.asarray(scaler.scale_)
        means = np.asarray(scaler.mean_)
        coef_z[:num_len] = coef_z[:num_len] / scales
        intercept -= float(
            np.dot(coef_z[:num_len], means)
        )  # adjust for means now that we changed coefs

    # Coefficients for other blocks remain as-is (OHE/booleans/geo_knn are already in "original" interpr.)
    return feat_names, coef_z, intercept


def extract_best_model_weights(best_pipe: Pipeline) -> Dict[str, Any]:
    """
    Extract interpretable weights/params from the best fitted pipeline.
    - For linear models (Linear/Ridge/Lasso): true final weights in original-input space (log-price), and multiplicative factors in price space.
    - For tree ensembles: labeled feature_importances_.
    - For others: fall back to get_params() (compact).
    Always includes 'estimator_class'.
    """
    out: Dict[str, Any] = {}

    # Unwrap: Pipeline(preprocess, regressor=TTR(regressor=<base>))
    reg = best_pipe.named_steps.get("regressor")
    base_reg = getattr(
        reg, "regressor", reg
    )  # unwrap TransformedTargetRegressor if present
    out["estimator_class"] = type(base_reg).__name__

    # Feature names from the fitted preprocessor
    pre: ColumnTransformer = best_pipe.named_steps["preprocess"]
    feat_names = list(pre.get_feature_names_out())

    # Linear models: map to original-input space
    if hasattr(base_reg, "coef_") and hasattr(base_reg, "intercept_"):
        w_z = np.asarray(base_reg.coef_).ravel()
        b_z = float(base_reg.intercept_)

        names, w_orig, b_orig = _linear_weights_in_original_space(best_pipe, w_z, b_z)

        # Multiplicative effects in price space for +1 unit change: factor = exp(weight)
        mult = np.exp(w_orig)

        out["space"] = "log-price"
        out["feature_names"] = names
        out["final_weights_logspace_labeled"] = _pair_labels(w_orig, names, "weight")
        out["final_effect_multipliers_price_space_labeled"] = _pair_labels(
            mult, names, "multiplier_per_unit"
        )
        out["intercept_log_price"] = b_orig
        out["note"] = (
            "Prediction is price = exp( intercept_log_price + sum_i weight_i * x_i + ... ). "
            "Multipliers are exp(weight_i): the factor by which price changes for +1 unit in feature i."
        )
        return out

    # Tree ensembles: feature_importances_ with labels
    if hasattr(base_reg, "feature_importances_"):
        fi = np.asarray(base_reg.feature_importances_)
        out["feature_names"] = feat_names
        out["feature_importances_labeled"] = _pair_labels(fi, feat_names, "importance")
        out["feature_importances_"] = _to_jsonable(fi)
        out["note"] = (
            "Tree ensembles do not have coefficients; importances reflect relative contribution."
        )
        return out

    # Fallback: compact hyperparameters
    try:
        params = base_reg.get_params(deep=False)
        out["params"] = {k: _to_jsonable(v) for k, v in params.items()}
    except Exception:
        out["params"] = {"detail": "unavailable"}
    out["note"] = (
        "Estimator has no explicit coefficients; stored hyperparameters instead."
    )
    return out


def distill_linear_surrogate(
    best_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Fit a simple Ridge on the pipeline's transformed features to approximate the BEST model
    in LOG-price space, then map coefficients back to original inputs.
    Returns labeled surrogate 'final weights' with an R² score vs the best model's log-preds.
    """
    pre: ColumnTransformer = best_pipe.named_steps["preprocess"]
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)
    names = list(pre.get_feature_names_out())

    # Target is log of the best model's price predictions (strictly positive)
    y_log_train = np.log(np.clip(y_pred_train, 1e-12, None))
    y_log_test = np.log(np.clip(y_pred_test, 1e-12, None))

    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(Z_train, y_log_train)

    # Quality of the surrogate
    r2_sur = float(r2_score(y_log_test, ridge.predict(Z_test)))

    # Map weights back to original input space
    w_z = ridge.coef_.ravel()
    b_z = float(ridge.intercept_)
    names, w_orig, b_orig = _linear_weights_in_original_space(best_pipe, w_z, b_z)
    mult = np.exp(w_orig)

    return {
        "surrogate_type": "Ridge",
        "r2_against_best_logspace": r2_sur,
        "feature_names": names,
        "final_weights_logspace_labeled": _pair_labels(w_orig, names, "weight"),
        "final_effect_multipliers_price_space_labeled": _pair_labels(
            mult, names, "multiplier_per_unit"
        ),
        "intercept_log_price": b_orig,
        "note": (
            "Surrogate approximates the best model in log-price space. "
            "Multipliers are exp(weight_i): factor on price for +1 unit of feature i."
        ),
    }


# ------------------- Core: train & save best -------------------
def train_and_save_best(
    df: pd.DataFrame,
    target: str = "price_chf",
    out_dir: str = ARTIFACT_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains many regressors, evaluates on a held-out test split, saves ONLY the best
    pipeline (.joblib), and writes run_meta.json with full per-model test stats.

    Enforces strictly positive predictions by training regressors in log-space
    using TransformedTargetRegressor (inverse = exp). Preprocessor and
    KNNLocalPrice still see the original y during their own fitting.
    Also stores the best model's *final* labeled weights. If the best is not linear,
    includes a linear surrogate's final weights as well.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"📂 Artifacts dir: {out_dir}")
    print(f"🎯 Target: {target}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    # Optionally drop forbidden columns if present
    df = drop_forbidden(df)

    X, y = df.drop(columns=[target]), df[target]

    # Split BEFORE fitting any preprocessing to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Strict inverse pair: works because y > 0 is guaranteed
    pos_y = FunctionTransformer(func=np.log, inverse_func=np.exp, validate=False)

    # Helper to wrap regressors
    def ttr(reg):
        return TransformedTargetRegressor(regressor=reg, transformer=pos_y)

    # Candidate models (trained in log-space; predictions inverse-transformed to > 0)
    models = {
        "Linear Regression": ttr(LinearRegression()),
        "Ridge Regression": ttr(Ridge(alpha=1.0)),
        "Lasso Regression": ttr(Lasso(alpha=0.001, max_iter=100000)),
        "Decision Tree": ttr(DecisionTreeRegressor(random_state=random_state)),
        "Random Forest": ttr(
            RandomForestRegressor(n_estimators=300, random_state=random_state)
        ),
        "Extra Trees": ttr(
            ExtraTreesRegressor(n_estimators=300, random_state=random_state)
        ),
        "Gradient Boosting": ttr(
            GradientBoostingRegressor(n_estimators=300, random_state=random_state)
        ),
        "KNN Regressor": ttr(KNeighborsRegressor(n_neighbors=5)),
        "SVR (RBF Kernel)": ttr(SVR(kernel="rbf", C=100, gamma=0.1)),
        "MLP Regressor": ttr(
            MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=5000, random_state=random_state
            )
        ),
    }

    results = []  # collect stats for all models
    best_name, best_pipe, best_rmse = None, None, float("inf")

    for idx, (name, est) in enumerate(models.items(), start=1):
        print(f"\n🚀 [{idx}/{len(models)}] Training {name}...")
        # Build a fresh preprocessor per pipeline so it's fit ONLY on train
        pre = build_preprocessor(X)
        pipe = Pipeline([("preprocess", pre), ("regressor", est)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)  # already inverse-transformed to strictly > 0
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
    safe_name = best_name.replace(" ", "_")
    model_path = os.path.join(out_dir, f"best_model_{safe_name}_{ts}.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\n💾 Saved best model: {best_name} (RMSE={best_rmse:.4f}) → {model_path}")

    # Predictions for surrogate (if needed)
    y_pred_train = best_pipe.predict(X_train)
    y_pred_test = best_pipe.predict(X_test)

    # Extract final weights/params for the BEST fitted model
    best_weights = extract_best_model_weights(best_pipe)
    best_weights["model_label"] = best_name  # human-readable label

    # If best is NOT linear, also provide a linear surrogate's final weights
    linear_classes = {"LinearRegression", "Ridge", "Lasso"}
    if best_weights.get("estimator_class") not in linear_classes:
        surrogate = distill_linear_surrogate(
            best_pipe, X_train, X_test, y_pred_train, y_pred_test, alpha=1.0
        )
        best_weights["surrogate_final_weights"] = surrogate

    # Save full run metadata with all per-model test stats + weights/params
    meta = {
        "target": target,
        "random_state": random_state,
        "test_size": test_size,
        "created_at": ts,
        "best_model": best_name,
        "best_test_rmse": float(best_rmse),
        "model_path": model_path,
        "all_results": results,  # full leaderboard-like stats
        "best_model_weights": best_weights,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata saved → {os.path.join(out_dir, 'run_meta.json')}")
    print(f"🔎 best_model_weights keys: {list(best_weights.keys())}")

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
    train_and_save_best(df, target="price_chf", out_dir=ARTIFACT_DIR)
