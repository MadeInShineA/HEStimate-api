from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace

app = FastAPI()


# ---------- Models
class VerifyRequest(BaseModel):
    image: str


class CompareRequest(BaseModel):
    image1: str
    image2: str


class VerifyResponse(BaseModel):
    success: bool
    message: str


# ---------- Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the HEStimate API, please check /docs for the doc"}


@app.post("/verify", status_code=200, response_model=VerifyResponse)
async def verify(payload: VerifyRequest):
    faces = DeepFace.extract_faces(img_path=payload.image, anti_spoofing=True)
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


@app.post("/compare", status_code=200)
async def compare(payload: CompareRequest) -> bool:
    try:
        result = DeepFace.verify(img1_path=payload.image1, image2=payload.image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")
    return result["verified"]
