from typing import List, Union, Optional, Literal
from fastapi import APIRouter, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import os, hmac


router = APIRouter()

ARTIFACT_DIR = Path("price_prediction/artifacts")
PRICE_MODEL = None  # cached pipeline
PRICE_MODEL_NAME: Optional[str] = None


def _find_latest_model() -> Path:
    if not ARTIFACT_DIR.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {ARTIFACT_DIR}")
    # Prefer live model if exists, else latest timestamped best_model_*.joblib
    live = ARTIFACT_DIR / "best_model_live.joblib"
    if live.exists():
        return live
    models = sorted(ARTIFACT_DIR.glob("best_model_*.joblib"))
    if not models:
        raise FileNotFoundError(
            f"No best_model_*.joblib found in {ARTIFACT_DIR}. Train and export a model first."
        )
    return models[-1]


def get_price_model():
    global PRICE_MODEL, PRICE_MODEL_NAME
    model_path = _find_latest_model()
    if PRICE_MODEL is None or PRICE_MODEL_NAME != model_path.name:
        PRICE_MODEL = joblib.load(model_path)
        PRICE_MODEL_NAME = model_path.name
    return PRICE_MODEL, PRICE_MODEL_NAME


# ---------------- Helper: get geo_knn transformer from pipeline ----------------
def _get_geo_knn_transformer(pipe):
    pre = pipe.named_steps.get("preprocess")
    if pre is None or not hasattr(pre, "named_transformers_"):
        raise HTTPException(
            status_code=500, detail="Preprocessor not found in pipeline."
        )
    geo_knn = pre.named_transformers_.get("geo_knn")
    if geo_knn is None:
        raise HTTPException(
            status_code=500, detail="geo_knn transformer not found in pipeline."
        )
    return geo_knn


def _load_allowed_keys() -> List[str]:
    raw = os.getenv("OBS_TOKENS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def verify_key(x_obs_token: Optional[str] = Header(default=None, alias="X-OBS-TOKEN")):
    allowed = _load_allowed_keys()
    if not x_obs_token or not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token."
        )
    for k in allowed:
        if hmac.compare_digest(x_obs_token, k):
            return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token."
    )


# ---------------- Schemas (Price Prediction) ----------------
# NOTE: No nearest_hesso_name here; proxim_hesso_km is kept
class EstimatePriceRequest(BaseModel):
    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude en degrés (-90 à 90)"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude en degrés (-180 à 180)"
    )
    surface_m2: float = Field(gt=0, description="Surface en m² (>0)")
    num_rooms: int = Field(gt=0, description="Nombre de pièces (>0)")
    type: Literal["room", "entire_home"] = "room"
    is_furnished: bool
    floor: int = Field(ge=0, description="Étage (>=0)")
    wifi_incl: bool
    charges_incl: bool
    car_park: bool
    dist_public_transport_km: float = Field(
        gt=0, description="Distance transport public en km (>0)"
    )
    proxim_hesso_km: float = Field(gt=0, description="Distance HES en km (>0)")


class EstimatePriceItemResponse(BaseModel):
    predicted_price_chf: float
    model_artifact: str


# ------- Live updates for geo KNN -------
class LabeledObservation(BaseModel):
    latitude: float
    longitude: float
    price_chf: float
    observed_at: Optional[str] = Field(
        default=None,
        description="Optional ISO datetime; not required (for future recency weighting).",
    )


class UpdateResponse(BaseModel):
    success: bool
    message: str
    added: int
    model_artifact: str


class ModelInfo(BaseModel):
    artifact: str
    geo_points: int


@router.post(
    "/estimate-price",
    response_model=Union[EstimatePriceItemResponse, List[EstimatePriceItemResponse]],
)
async def estimate_price(
    payload: Union[EstimatePriceRequest, List[EstimatePriceRequest]],
):
    try:
        pipe, model_name = get_price_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # Normalize to list of dicts
    if isinstance(payload, list):
        records = [p.model_dump() for p in payload]
    else:
        records = [payload.model_dump()]

    df = pd.DataFrame.from_records(records)
    try:
        preds = pipe.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    results = [
        EstimatePriceItemResponse(
            predicted_price_chf=float(round(p, 2)),
            model_artifact=model_name,
        )
        for p in preds
    ]
    return results if isinstance(payload, list) else results[0]


# ---- Price: add labeled observations to improve geo-KNN ----
@router.post(
    "/observations", dependencies=[Depends(verify_key)], response_model=UpdateResponse
)
async def add_observations(items: List[LabeledObservation]):
    if not items:
        raise HTTPException(status_code=400, detail="Empty payload.")
    try:
        pipe, _ = get_price_model()
        geo_knn = _get_geo_knn_transformer(pipe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    lat = np.array([it.latitude for it in items], dtype=float)
    lon = np.array([it.longitude for it in items], dtype=float)
    y = np.array([it.price_chf for it in items], dtype=float)
    X_ll_new = np.stack([lat, lon], axis=1)

    try:
        if not hasattr(geo_knn, "update"):
            raise RuntimeError(
                "geo_knn transformer is not updatable; retrain with live-updatable KNNLocalPrice."
            )
        geo_knn.update(X_ll_new, y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Update failed: {e}")

    # Persist a live copy so improvements survive restarts

    live_path = ARTIFACT_DIR / "best_model_live.joblib"
    try:
        joblib.dump(pipe, live_path)
        global PRICE_MODEL, PRICE_MODEL_NAME
        PRICE_MODEL = pipe
        PRICE_MODEL_NAME = live_path.name
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not persist updated model: {e}"
        )

    return UpdateResponse(
        success=True,
        message="Observations added and model updated.",
        added=len(items),
        model_artifact=live_path.name,
    )


# ---- Price: info ----
@router.get("/model-info", response_model=ModelInfo)
async def model_info():
    try:
        pipe, model_name = get_price_model()
        geo_knn = _get_geo_knn_transformer(pipe)
        total = 0 if getattr(geo_knn, "_xy", None) is None else len(geo_knn._xy)
        live_only = max(0, total)
        return ModelInfo(artifact=model_name, geo_points=live_only)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {e}")
