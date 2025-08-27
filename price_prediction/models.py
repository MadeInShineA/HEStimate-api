# ==============================================================
# Train & Compare MANY Models + CV + Geo/KNN Feature + Target Enc
#   - Robust to unknown cities
#   - Live-updatable KNNLocalPrice (API can learn from new labeled points)
#   - Saves artifacts under price-prediction/artifacts/
# ==============================================================

import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Dict, Tuple, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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


# ----------------- Custom transformers ----------------


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Cross-validated target encoder with smoothing.
    Unseen categories map to the global mean. When inside a Pipeline, each CV fold
    fits its own encoder, preventing leakage.
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
      - Target-encode: city (unseen -> global mean)
      - OneHot: postal_code, type
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
    default_cat_ohe = ["postal_code", "type"]
    default_te = ["city"]  # ONLY city; nearest_hesso_name removed
    default_bool = ["is_furnished", "wifi_incl", "charges_incl", "car_park"]

    num_features = [c for c in default_num if c in X.columns]
    cat_ohe_features = [c for c in default_cat_ohe if c in X.columns]
    te_features = [c for c in default_te if c in X.columns]
    bool_features = [c for c in default_bool if c in X.columns]

    ohe = _make_ohe()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "te_city",
                CVTargetEncoder(te_features, n_splits=5, smoothing=50, random_state=42),
                te_features,
            ),
            ("cat", ohe, cat_ohe_features),
            ("bool", "passthrough", bool_features),
            ("geo_knn", KNNLocalPrice(n_neighbors=10), ["latitude", "longitude"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    out_names = trans.get_feature_names_out(
                        cols if not isinstance(cols, list) else cols
                    )
                    names.extend(list(out_names))
                    continue
                except Exception:
                    pass
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend(cols)
            else:
                names.append(str(cols))
        return np.array(names, dtype=object)


def cv_scores(
    pipeline: Pipeline, X, y, cv_splits=5, random_state=42
) -> Dict[str, float]:
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    mae = -cross_val_score(
        pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv
    ).mean()
    rmse = math.sqrt(
        -cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=cv).mean()
    )
    r2 = cross_val_score(pipeline, X, y, scoring="r2", cv=cv).mean()
    return {"CV_MAE": mae, "CV_RMSE": rmse, "CV_R2": r2}


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test) -> Dict[str, float]:
    preds = pipeline.predict(X_test)
    return {
        "Test_MAE": mean_absolute_error(y_test, preds),
        "Test_RMSE": math.sqrt(mean_squared_error(y_test, preds)),
        "Test_R2": r2_score(y_test, preds),
    }


def plot_pred_vs_actual(y_true, y_pred, title: str, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    ]
    plt.plot(lims, lims)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_importance(pipeline: Pipeline, out_dir: str, model_name: str):
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["regressor"]
    feat_names = get_feature_names(pre)

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {"feature": feat_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi.to_csv(
            os.path.join(
                out_dir, f"feature_importance_{model_name.replace(' ', '_')}.csv"
            ),
            index=False,
        )
    if isinstance(model, (LinearRegression, Ridge, Lasso)) and hasattr(model, "coef_"):
        coef = pd.DataFrame(
            {"feature": feat_names, "coefficient": model.coef_}
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
        coef.to_csv(
            os.path.join(
                out_dir, f"linear_coefficients_{model_name.replace(' ', '_')}.csv"
            ),
            index=False,
        )


# ------------------- Main training -------------------


def train_compare_and_save(
    df: pd.DataFrame,
    target: str = "price_chf",
    out_dir: str = ARTIFACT_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"📂 Artifacts: {out_dir}")
    print(f"🎯 Target: {target}")

    X, y = df.drop(columns=[target]), df[target]

    # Build & preview preprocessor
    pre = build_preprocessor(X)
    pre.fit(X, y)
    print(f"✅ Preprocessor ready with {len(get_feature_names(pre))} output features.")

    # Models
    models = {
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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

    results, fitted = [], {}
    for idx, (name, estimator) in enumerate(models.items(), 1):
        print(f"\n🚀 [{idx}/{len(models)}] Training {name}...")
        pipe = Pipeline([("preprocess", pre), ("regressor", estimator)])
        pipe.fit(X_train, y_train)
        print("   ✅ Trained.")

        # Test metrics
        print("   📈 Evaluating on test...")
        test_metrics = evaluate_pipeline(pipe, X_test, y_test)
        print(f"   🔢 Test: {test_metrics}")

        # CV metrics
        print(f"   🔄 {cv_splits}-fold CV...")
        cv_metrics = cv_scores(
            Pipeline([("preprocess", pre), ("regressor", estimator)]),
            X,
            y,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        print(f"   🔢 CV: {cv_metrics}")

        results.append({"Model": name, **test_metrics, **cv_metrics})
        fitted[name] = pipe
        save_feature_importance(pipe, out_dir, name)

    # Leaderboard
    leaderboard = pd.DataFrame(results).sort_values(by="Test_RMSE", ascending=True)
    print("\n=== 🏆 Leaderboard (by Test_RMSE) ===")
    print(leaderboard.to_string(index=False))
    lb_path = os.path.join(out_dir, "leaderboard.csv")
    leaderboard.to_csv(lb_path, index=False)
    print(f"💾 Leaderboard saved: {lb_path}")

    # Best model
    best_name = leaderboard.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    preds = best_pipe.predict(X_test)
    plot_path = os.path.join(
        out_dir, f"pred_vs_actual_{best_name.replace(' ', '_')}.png"
    )
    plot_pred_vs_actual(y_test, preds, f"Predicted vs Actual — {best_name}", plot_path)
    print(f"📉 Plot saved: {plot_path}")

    # Save pipeline
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        out_dir, f"best_model_{best_name.replace(' ', '_')}_{ts}.joblib"
    )
    joblib.dump(best_pipe, model_path)
    print(f"💾 Best model saved: {model_path}")

    # Save metadata
    meta = {
        "target": target,
        "random_state": random_state,
        "test_size": test_size,
        "cv_splits": cv_splits,
        "best_model": best_name,
        "created_at": ts,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return fitted, leaderboard


# ----------------------- CLI -----------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"❌ CSV not found: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        train_compare_and_save(df, target="price_chf", out_dir=ARTIFACT_DIR)
    else:
        print("Usage: python train_models.py /path/to/data.csv")
