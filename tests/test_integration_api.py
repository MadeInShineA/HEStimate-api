import pytest
from fastapi.testclient import TestClient
import base64
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from main import app


client = TestClient(app)

def load_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------- /verify endpoint
@pytest.mark.integration
def test_verify_real_face():
    image_base64 = load_image_as_base64("tests/images/olivier.jpg")
    resp = client.post("/verify", json={"image": image_base64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "Face verified" in data["message"]

@pytest.mark.integration
def test_verify_no_face_detected():
    image_base64 = load_image_as_base64("tests/images/nothing.jpg")
    resp = client.post("/verify", json={"image": image_base64})
    assert resp.status_code == 400
    data = resp.json()
    assert "No face detected" in data["detail"]

# ---------- /compare endpoint
@pytest.mark.integration
def test_compare_same_person():
    img1_b64 = load_image_as_base64("tests/images/olivier.jpg")
    img2_b64 = load_image_as_base64("tests/images/olivier2.jpg")
    resp = client.post("/compare", json={"image1": img1_b64, "image2": img2_b64})
    assert resp.status_code == 200
    assert resp.json() is True

@pytest.mark.integration
def test_compare_different_persons():
    img1_b64 = load_image_as_base64("tests/images/simon.jpg")
    img2_b64 = load_image_as_base64("tests/images/olivier.jpg")
    resp = client.post("/compare", json={"image1": img1_b64, "image2": img2_b64})
    assert resp.status_code == 200
    assert resp.json() is False
