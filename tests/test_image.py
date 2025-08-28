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


# ---------- b64_to_ndarray with data URI prefix
@pytest.mark.unit
def test_b64_to_ndarray_with_data_uri_prefix():
    from main import b64_to_ndarray

    # prepare a tiny 1x1 black PNG image
    import cv2
    import numpy as np

    img = np.zeros((1, 1, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buf).decode("utf-8")
    data_uri_b64 = f"data:image/png;base64,{img_b64}"

    # call the function
    result = b64_to_ndarray(data_uri_b64)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1, 3)


@pytest.mark.unit
def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "Welcome to the HEStimate API" in data["message"]


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


@pytest.mark.integration
def test_verify_two_faces():
    image_base64 = load_image_as_base64("tests/images/two-faces.jpg")
    resp = client.post("/verify", json={"image": image_base64})
    assert resp.status_code == 400
    data = resp.json()
    assert "Please take a picture with exactly 1 person" in data["detail"]


@pytest.mark.integration
def test_verify_spoofing():
    image_base64 = load_image_as_base64("tests/images/spoofing.jpg")
    resp = client.post("/verify", json={"image": image_base64})
    assert resp.status_code == 400
    data = resp.json()
    assert "Spoofing detected. Please provide a real face image." in data["detail"]


@pytest.mark.unit
def test_verify_invalid_base64_rejected():
    # clearly invalid base64
    bad_b64 = "this-is-not-base64!!"
    resp = client.post("/verify", json={"image": bad_b64})
    assert resp.status_code == 400
    assert "Invalid base64 image data" in resp.json()["detail"]


@pytest.mark.unit
def test_verify_could_not_decode_image():
    # valid base64, but not an image -> cv2.imdecode returns None
    not_image_b64 = base64.b64encode(b"hello world").decode("utf-8")
    resp = client.post("/verify", json={"image": not_image_b64})
    assert resp.status_code == 400
    assert "Could not decode image" in resp.json()["detail"]


# ---------- /compare endpoint
@pytest.mark.integration
def test_compare_same_person():
    img1_b64 = load_image_as_base64("tests/images/olivier.jpg")
    img2_b64 = load_image_as_base64("tests/images/olivier2.jpg")
    resp = client.post("/compare", json={"image1": img1_b64, "image2": img2_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "Face verified and appears real" == data["message"]


@pytest.mark.integration
def test_compare_different_persons():
    img1_b64 = load_image_as_base64("tests/images/simon.jpg")
    img2_b64 = load_image_as_base64("tests/images/olivier.jpg")
    resp = client.post("/compare", json={"image1": img1_b64, "image2": img2_b64})
    assert resp.status_code == 400
    data = resp.json()
    assert "Faces do not match" in data["detail"]
