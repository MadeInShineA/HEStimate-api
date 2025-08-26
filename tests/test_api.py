import importlib
import sys
import types
import pytest
from fastapi.testclient import TestClient


# Base64 pour une image PNG 1x1 (toujours décodable)
ONE_BY_ONE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAgMBg6Q6iOQAAAAASUVORK5CYII="
)

def load_app_with_fake_deepface(monkeypatch, faces=None, boom_in_extract=False, verify_impl=None):
    # Fake deepface module
    fake_deepface = types.SimpleNamespace()

    def fake_extract(img, **kwargs):
        if boom_in_extract:
            raise Exception("boom in extract")
        return faces or []

    def fake_verify(img1, img2, **kwargs):
        if verify_impl:
            return verify_impl(img1, img2, **kwargs)
        return {"verified": False}

    fake_deepface.DeepFace = types.SimpleNamespace(
        extract_faces=fake_extract,
        verify=fake_verify,
    )

    # ⚡ clean sys.modules
    sys.modules.pop("deepface", None)
    sys.modules["deepface"] = fake_deepface
    sys.modules.pop("main", None)

    # Re-import main with fake deepface
    import main
    importlib.reload(main)

    return main.app, fake_deepface


# ---------- cleanup après chaque test ----------
def teardown_function():
    sys.modules.pop("deepface", None)
    sys.modules.pop("main", None)


# ---------- root
def test_root_returns_welcome(monkeypatch):
    app, _ = load_app_with_fake_deepface(monkeypatch)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {
        "message": "Welcome to the HEStimate API, please check /docs for the doc"
    }


# ---------- /verify
@pytest.mark.parametrize(
    "faces, expected_status, expected_json",
    [
        (
            [{"is_real": True}, {"is_real": True}],
            400,
            {"detail": "Please take a picture with exactly 1 person"},
        ),
        ([], 400, {"detail": "Please take a picture with exactly 1 person"}),
        (
            [{"is_real": False}],
            400,
            {"detail": "Spoofing detected. Please provide a real face image."},
        ),
        (
            [{"is_real": None}],
            400,
            {"detail": "Anti-spoofing check unavailable for this image."},
        ),
        (
            [{"is_real": True}],
            200,
            {"success": True, "message": "Face verified and appears real"},
        ),
    ],
)
def test_verify_all_responses(monkeypatch, faces, expected_status, expected_json):
    app, _ = load_app_with_fake_deepface(monkeypatch, faces=faces)
    client = TestClient(app)

    resp = client.post("/verify", json={"image": ONE_BY_ONE_PNG_B64})
    assert resp.status_code == expected_status
    assert resp.json() == expected_json


def test_verify_unexpected_exception_returns_500(monkeypatch):
    app, _ = load_app_with_fake_deepface(monkeypatch, boom_in_extract=True)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/verify", json={"image": ONE_BY_ONE_PNG_B64})
    assert resp.status_code == 500


# ---------- /compare
def test_compare_returns_true(monkeypatch):
    def fake_verify(img1_path, img2_path, **kwargs):
        return {"verified": True}

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        "/compare", json={"image1": ONE_BY_ONE_PNG_B64, "image2": ONE_BY_ONE_PNG_B64}
    )
    assert resp.status_code == 200
    assert resp.json() is True


def test_compare_returns_false(monkeypatch):
    def fake_verify(img1_path, img2_path, **kwargs):
        return {"verified": False}

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        "/compare", json={"image1": ONE_BY_ONE_PNG_B64, "image2": ONE_BY_ONE_PNG_B64}
    )
    assert resp.status_code == 200
    assert resp.json() is False


def test_compare_handles_error(monkeypatch):
    def fake_verify(img1_path, img2_path, **kwargs):
        raise ValueError("bad input")

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        "/compare", json={"image1": ONE_BY_ONE_PNG_B64, "image2": ONE_BY_ONE_PNG_B64}
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Comparison failed: bad input"}
