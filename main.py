# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from pathlib import Path

import base64
import numpy as np
import pandas as pd
import cv2
import joblib

from deepface import DeepFace

import sys, types

from price_prediction.models import CVTargetEncoder as _LegacyCVTargetEncoder
from price_prediction.models import KNNLocalPrice as _LegacyKNNLocalPrice

# ---------------- App ----------------
app = FastAPI(title="HEStimate API")


# ---------------- Utils: images ----------------
def b64_to_ndarray(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


_main_mod = sys.modules.get("__main__")
if _main_mod is None:
    _main_mod = types.ModuleType("__main__")
    sys.modules["__main__"] = _main_mod

setattr(_main_mod, "CVTargetEncoder", _LegacyCVTargetEncoder)
setattr(_main_mod, "KNNLocalPrice", _LegacyKNNLocalPrice)


# ---------------- Schemas (Face) ----------------
class VerifyRequest(BaseModel):
    image: str


class CompareRequest(BaseModel):
    image1: str
    image2: str


class VerifyResponse(BaseModel):
    success: bool
    message: str


class CompareResponse(BaseModel):
    success: bool
    message: str


# ---------------- Price model: artifact loading ----------------
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
    if PRICE_MODEL is None:
        model_path = _find_latest_model()
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


# ---------------- Schemas (Price Prediction) ----------------
# NOTE: No nearest_hesso_name here; proxim_hesso_km is kept
class EstimatePriceRequest(BaseModel):
    latitude: float
    longitude: float
    surface_m2: float
    num_rooms: float
    type: str
    is_furnished: bool
    floor: int
    wifi_incl: bool
    charges_incl: bool
    car_park: bool
    dist_public_transport_km: float
    proxim_hesso_km: float


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


# ---------------- Routes ----------------
@app.get("/")
async def root():
    return {"message": "Welcome to the HEStimate API, please check /docs for the doc"}


# ---- Face: verify ----
@app.post("/verify", status_code=200, response_model=VerifyResponse)
async def verify(payload: VerifyRequest):
    img = b64_to_ndarray(payload.image)
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            enforce_detection=True,
            align=True,
            anti_spoofing=True,
            detector_backend="yolov11m"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No face detected: {e}")

    if len(faces) != 1:
        raise HTTPException(
            status_code=400, detail="Please take a picture with exactly 1 person"
        )
    elif faces[0].get("is_real") is False:
        raise HTTPException(
            status_code=400,
            detail="Spoofing detected. Please provide a real face image.",
        )
    elif faces[0].get("is_real") is None:
        raise HTTPException(
            status_code=400, detail="Anti-spoofing check unavailable for this image."
        )
    else:
        return {"success": True, "message": "Face verified and appears real"}


# ---- Face: compare ----
@app.post("/compare", status_code=200, response_model=CompareResponse)
async def compare(payload: CompareRequest):
    img1 = b64_to_ndarray(payload.image1)
    img2 = b64_to_ndarray(payload.image2)
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            detector_backend="yolov11m",
            model_name="ArcFace",
            enforce_detection=True,
            align=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")

    if not result.get("verified"):
        raise HTTPException(status_code=400, detail="Faces do not match")
    return {"success": True, "message": "Face verified and appears real"}


# ---- Price: estimate ----
@app.post(
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
        records = [p.dict() for p in payload]
    else:
        records = [payload.dict()]

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
@app.post("/observations", response_model=UpdateResponse)
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
@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    try:
        pipe, model_name = get_price_model()
        geo_knn = _get_geo_knn_transformer(pipe)
        total = 0 if getattr(geo_knn, "_xy", None) is None else len(geo_knn._xy)
        n_baseline = getattr(geo_knn, "_n_baseline", total)
        live_only = max(0, total - n_baseline)
        return ModelInfo(artifact=model_name, geo_points=live_only)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {e}")
