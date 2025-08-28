from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import base64
import numpy as np
import cv2


router = APIRouter()

# ---------- Models
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


@router.post("/verify", status_code=200, response_model=VerifyResponse)
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
    elif faces[0]["is_real"] is False:
        raise HTTPException(
            status_code=400,
            detail="Spoofing detected. Please provide a real face image.",
        )
    elif faces[0]["is_real"] is None:
        raise HTTPException(
            status_code=400, detail="Anti-spoofing check unavailable for this image."
        )
    else:
        return {"success": True, "message": "Face verified and appears real"}


@router.post("/compare", status_code=200, response_model=CompareResponse)
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

    if not result["verified"]:
        raise HTTPException(status_code=400, detail="Faces do not match")
    return {"success": True, "message": "Face verified and appears real"}
