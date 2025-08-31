# price_prediction/api.py

from __future__ import annotations

from typing import List, Union, Optional, Literal, Tuple
from fastapi import APIRouter, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import os
import hmac
from threading import RLock


router = APIRouter()

ARTIFACT_DIR = Path("price_prediction/artifacts")
LIVE_NAME = "best_model_live.joblib"

# ----------------- In-memory cache + locking -----------------
PRICE_MODEL = None  # cached pipeline object
PRICE_MODEL_NAME: Optional[str] = None  # file name
PRICE_MODEL_MTIME: Optional[float] = None  # file modification time
PRICE_MODEL_SIZE: Optional[int] = None  # file size
_MODEL_LOCK = RLock()


# ----------------- Security (API key) -----------------
def _load_allowed_keys() -> List[str]:
    raw = os.getenv("API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def verify_key(api_key: Optional[str] = Header(default=None, alias="API-KEY")):
    allowed = _load_allowed_keys()
    if not api_key or not allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token."
        )
    for k in allowed:
        if hmac.compare_digest(api_key, k):
            return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token."
    )


# ----------------- Models / Schemas -----------------
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
class ObservationRequest(BaseModel):
    latitude: float
    longitude: float
    price_chf: float


class UpdateResponse(BaseModel):
    success: bool
    message: str
    added: int
    model_artifact: str


class ModelInfo(BaseModel):
    artifact: str
    geo_points: int


# ----------------- Internal helpers -----------------
def _stat(p: Path) -> Tuple[float, int]:
    s = p.stat()
    return s.st_mtime, s.st_size


def _seed_live_if_needed() -> Path:
    """
    Ensure artifacts dir exists and a live model file is present.
    If live file is missing, seed it from the latest best_model_*.joblib.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    live = ARTIFACT_DIR / LIVE_NAME
    if live.exists():
        return live

    candidates = sorted(ARTIFACT_DIR.glob("best_model_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"No artifacts to seed {LIVE_NAME}. Train/export at least once."
        )
    src = candidates[-1]
    tmp = ARTIFACT_DIR / (".best_model_live.joblib.tmp")
    # Copy bytes atomically
    with open(src, "rb") as r, open(tmp, "wb") as w:
        w.write(r.read())
    os.replace(tmp, live)
    return live


def _live_model_path() -> Path:
    # Always use the live file; create it if missing
    return _seed_live_if_needed()


def get_price_model():
    """
    Always use artifacts/best_model_live.joblib and hot-reload when the file changes.
    Uses filename + (mtime, size) to decide reload.
    """
    global PRICE_MODEL, PRICE_MODEL_NAME, PRICE_MODEL_MTIME, PRICE_MODEL_SIZE
    p = _live_model_path()

    try:
        m, sz = _stat(p)
    except FileNotFoundError:
        # If it was removed concurrently, re-seed and force reload
        p = _seed_live_if_needed()
        m, sz = _stat(p)

    need_reload = (
        PRICE_MODEL is None
        or PRICE_MODEL_NAME != p.name
        or PRICE_MODEL_MTIME != m
        or PRICE_MODEL_SIZE != sz
    )

    if need_reload:
        # Guard against concurrent writes while reading
        with _MODEL_LOCK:
            # Double-check inside the lock
            m2, sz2 = _stat(p)
            if (
                PRICE_MODEL is None
                or PRICE_MODEL_NAME != p.name
                or PRICE_MODEL_MTIME != m2
                or PRICE_MODEL_SIZE != sz2
            ):
                model = joblib.load(p)
                PRICE_MODEL = model
                PRICE_MODEL_NAME = p.name
                PRICE_MODEL_MTIME, PRICE_MODEL_SIZE = m2, sz2

    return PRICE_MODEL, PRICE_MODEL_NAME


# ---------------- Helper: access geo_knn transformer ----------------
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


# ----------------- Routes -----------------
@router.post(
    "/estimate-price",
    response_model=Union[EstimatePriceItemResponse, List[EstimatePriceItemResponse]],
)
async def estimate_price(
    payload: Union[EstimatePriceRequest, List[EstimatePriceRequest]],
):
    """
    Predict price and (post-hoc) blend with local KNN mean so neighbors always influence.
    Alpha can be tuned via env var KNN_BLEND_ALPHA (default 0.15).
    y_hat = (1 - alpha) * model_pred + alpha * local_knn_mean
    """
    # Normalize to list of dicts
    if isinstance(payload, list):
        records = [p.model_dump() for p in payload]
    else:
        records = [payload.model_dump()]

    df = pd.DataFrame.from_records(records)

    with _MODEL_LOCK:
        try:
            pipe, model_name = get_price_model()
            # Base model prediction
            preds = pipe.predict(df).astype(float)

            # Post-hoc blend with local KNN mean (computed from the live, updatable geo_knn)
            pre = pipe.named_steps["preprocess"]
            geo = pre.named_transformers_["geo_knn"]
            local = geo.transform(df[["latitude", "longitude"]]).ravel().astype(float)

            alpha = float(os.getenv("KNN_BLEND_ALPHA", "0.15"))
            if alpha > 0.0:
                preds = (1.0 - alpha) * preds + alpha * local

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    results = [
        EstimatePriceItemResponse(
            predicted_price_chf=float(p),  # keep full precision
            model_artifact=model_name,
        )
        for p in preds
    ]
    return results if isinstance(payload, list) else results[0]


@router.post(
    "/observations",
    dependencies=[Depends(verify_key)],
    response_model=UpdateResponse,
)
async def add_observations(item: ObservationRequest):
    """
    Append labeled (lat, lon, price) observations to the live model's geo-KNN,
    then persist atomically to best_model_live.joblib so all workers can reload.
    """
    if not item:
        raise HTTPException(status_code=400, detail="Empty payload.")

    lat = np.array([item.latitude], dtype=float)
    lon = np.array([item.longitude], dtype=float)
    y = np.array([item.price_chf], dtype=float)
    X_ll_new = np.stack([lat, lon], axis=1)

    with _MODEL_LOCK:
        try:
            pipe, _ = get_price_model()
            geo_knn = _get_geo_knn_transformer(pipe)
            if not hasattr(geo_knn, "update"):
                raise RuntimeError(
                    "geo_knn transformer is not updatable; retrain with live-updatable KNNLocalPrice."
                )
            geo_knn.update(X_ll_new, y)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Update failed: {e}")

        # Persist atomically so other processes pick it up by mtime/size change
        live_path = ARTIFACT_DIR / LIVE_NAME
        tmp_path = ARTIFACT_DIR / (".best_model_live.joblib.tmp")
        try:
            joblib.dump(pipe, tmp_path)
            os.replace(tmp_path, live_path)  # atomic on POSIX
            # Update cached stats so first subsequent read can skip disk if same process
            global PRICE_MODEL_MTIME, PRICE_MODEL_SIZE
            PRICE_MODEL_MTIME, PRICE_MODEL_SIZE = _stat(live_path)
        except Exception as e:
            # Best-effort cleanup
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            raise HTTPException(
                status_code=500, detail=f"Could not persist updated model: {e}"
            )

    return UpdateResponse(
        success=True,
        message="Observations added and live model updated.",
        added=1,
        model_artifact=LIVE_NAME,
    )


@router.get("/model-info", response_model=ModelInfo)
async def model_info():
    with _MODEL_LOCK:
        try:
            pipe, model_name = get_price_model()
            geo_knn = _get_geo_knn_transformer(pipe)
            total = 0 if getattr(geo_knn, "_xy", None) is None else len(geo_knn._xy)
            return ModelInfo(artifact=model_name, geo_points=total)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model info error: {e}")
