from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace


app = FastAPI()


class VerifyResponse(BaseModel):
    success: bool
    message: str

@app.get('/')
async def root():
    return {'message': 'Welcome to the HEStimate API, please check /docs for the doc'}

@app.post('/verify', status_code=200, response_model=VerifyResponse)
async def verify(image: str):
    faces = DeepFace.extract_faces(img_path=image, anti_spoofing=True)
    if len(faces) != 1:
        raise HTTPException(status_code=400, detail="Please take a picture with exactly 1 person")
    elif faces[0]["is_real"] is False:
        raise HTTPException(status_code=400, detail="Spoofing detected. Please provide a real face image.")
    elif faces[0]["is_real"] is None:
        raise HTTPException(status_code=400, detail="Anti-spoofing check unavailable for this image.")
    else:
        return {"success": True, "message": "Face verified and appears real"}

@app.post('/compare', status_code=200)
async def compare(image1: str, image2: str) -> bool:
    try:
        result = DeepFace.verify(img1_path=image1, image2=image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")

    return result['verified']
