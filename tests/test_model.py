import pytest
from fastapi.testclient import TestClient
import sys
import os
import time

# Make the project importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app  # if your file isn't main.py, update this

client = TestClient(app)


# ---------- Helpers ----------


def valid_payload():
    """Matches EstimatePriceRequest schema (no nearest_hesso_name)."""
    return {
        "latitude": 46.2315,
        "longitude": 7.3606,
        "surface_m2": 20.0,
        "num_rooms": 1.0,
        "type": "room",
        "is_furnished": True,
        "floor": 2,
        "wifi_incl": True,
        "charges_incl": False,
        "car_park": False,
        "dist_public_transport_km": 0.25,
        "proxim_hesso_km": 1.2,
    }


def ensure_model_or_skip():
    """Hit /model-info; if the app can't load a model, skip these integration tests."""
    resp = client.get("/model-info")
    if resp.status_code != 200:
        # If model isn't available, skip to avoid hard failures in CI.
        detail = resp.json().get("detail", "")
        pytest.skip(f"Model not available for integration tests: {detail}")
    data = resp.json()
    # Some artifacts may load but lack geo_knn; that's okay for estimate tests.
    return data


# ---------- Tests ----------


@pytest.mark.integration
def test_model_info_integration():
    info = ensure_model_or_skip()
    assert "artifact" in info
    assert "geo_points" in info
    assert isinstance(info["geo_points"], int)
    # Artifact name will vary; just ensure it's a non-empty string
    assert isinstance(info["artifact"], str) and len(info["artifact"]) > 0


@pytest.mark.integration
def test_estimate_price_single_integration():
    ensure_model_or_skip()
    resp = client.post("/estimate-price", json=valid_payload())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "predicted_price_chf" in data
    assert isinstance(data["predicted_price_chf"], (int, float))
    assert "model_artifact" in data


@pytest.mark.integration
def test_estimate_price_batch_integration():
    ensure_model_or_skip()
    p1 = valid_payload()
    p2 = valid_payload()
    p2["surface_m2"] = 35.0  # small change
    resp = client.post("/estimate-price", json=[p1, p2])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list) and len(data) == 2
    for item in data:
        assert "predicted_price_chf" in item
        assert "model_artifact" in item


@pytest.mark.integration
def test_observations_empty_payload_400():
    ensure_model_or_skip()
    resp = client.post("/observations", json=[])
    assert resp.status_code == 400
    assert "Empty payload" in resp.json().get("detail", "")


@pytest.mark.integration
def test_observations_add_and_list_integration():
    """
    Add 2 labeled geo points, then:
      - /observations should include them
      - /model-info geo_points should increase by at least 2
    If the model lacks an updatable geo_knn, gracefully skip.
    """
    info_before = ensure_model_or_skip()
    geo_points_before = info_before.get("geo_points", 0)

    # Add two labeled observations
    add_payload = [
        {"latitude": 46.2315, "longitude": 7.3606, "price_chf": 1220.0},
        {"latitude": 46.1066, "longitude": 7.0707, "price_chf": 980.0},
    ]
    resp = client.post("/observations", json=add_payload)

    if resp.status_code == 500 and "not updatable" in resp.json().get("detail", ""):
        pytest.skip(
            "geo_knn transformer is not updatable in current artifact; retrain with live-updatable KNNLocalPrice."
        )
    elif resp.status_code == 500 and "geo_knn transformer not found" in resp.json().get(
        "detail", ""
    ):
        pytest.skip("geo_knn transformer not found in current artifact.")
    elif resp.status_code == 500 and "Model load error" in resp.json().get(
        "detail", ""
    ):
        pytest.skip("Model load error; artifact may be missing.")
    else:
        assert resp.status_code == 200, resp.text

    data_add = resp.json()
    assert data_add["success"] is True
    assert data_add["added"] == 2

    # Small wait to ensure any in-memory updates are visible (usually immediate)
    time.sleep(0.05)

    # /observations should list at least these two points (plus any existing)
    resp_list = client.get("/observations")
    assert resp_list.status_code == 200, resp_list.text
    listed = resp_list.json()
    assert isinstance(listed, list)

    # Check that both added points appear in the list
    def present(lat, lon, price):
        for item in listed:
            if (
                abs(item["latitude"] - lat) < 1e-6
                and abs(item["longitude"] - lon) < 1e-6
                and abs(item["price_chf"] - price) < 1e-6
            ):
                return True
        return False

    assert present(46.2315, 7.3606, 1220.0)
    assert present(46.1066, 7.0707, 980.0)

    # /model-info should reflect increased count
    resp_after = client.get("/model-info")
    assert resp_after.status_code == 200, resp_after.text
    info_after = resp_after.json()
    assert info_after["geo_points"] >= geo_points_before + 2
