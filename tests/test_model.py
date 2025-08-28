from fastapi.testclient import TestClient
from main import app
from dotenv import load_dotenv

import pytest
import os


client = TestClient(app)
load_dotenv()


# ------------------ Tests ------------------
@pytest.mark.integration
def test_estimate_price_single_real_model():
    payload = {
        "latitude": 46.5,
        "longitude": 6.5,
        "surface_m2": 50.0,
        "num_rooms": 2,
        "type": "room",
        "is_furnished": True,
        "floor": 1,
        "wifi_incl": True,
        "charges_incl": False,
        "car_park": True,
        "dist_public_transport_km": 0.5,
        "proxim_hesso_km": 1.2
    }
    resp = client.post("/estimate-price", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_price_chf" in data
    assert "model_artifact" in data

@pytest.mark.integration
def test_estimate_price_batch_real_model():
    payload = [
        {
            "latitude": 46.5, "longitude": 6.5, "surface_m2": 50.0,
            "num_rooms": 2, "type": "room", "is_furnished": True,
            "floor": 1, "wifi_incl": True, "charges_incl": False,
            "car_park": True, "dist_public_transport_km": 0.5,
            "proxim_hesso_km": 1.2
        },
        {
            "latitude": 47.0, "longitude": 7.0, "surface_m2": 70.0,
            "num_rooms": 3, "type": "entire_home", "is_furnished": False,
            "floor": 2, "wifi_incl": False, "charges_incl": True,
            "car_park": False, "dist_public_transport_km": 1.0,
            "proxim_hesso_km": 2.0
        }
    ]
    resp = client.post("/estimate-price", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2

@pytest.mark.integration
def test_add_observations_real_model():
    payload = [
        {"latitude": 46.5, "longitude": 6.5, "price_chf": 1500},
        {"latitude": 46.6, "longitude": 6.6, "price_chf": 1600}
    ]
    resp = client.post("/observations", json=payload, headers={"API_KEY": os.getenv('API_KEY')})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["added"] == 2
    assert "model_artifact" in data

@pytest.mark.integration
def test_model_info_real_model():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "artifact" in data
    assert "geo_points" in data

@pytest.mark.integration
def test_add_observations_increases_geo_points():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    initial_geo_points = resp.json()["geo_points"]

    new_obs = [
        {"latitude": 46.5, "longitude": 6.5, "price_chf": 1500},
        {"latitude": 46.6, "longitude": 6.6, "price_chf": 1600}
    ]
    resp_add = client.post("/observations", json=new_obs, headers={"API_KEY": os.getenv('API_KEY')})
    assert resp_add.status_code == 200
    data_add = resp_add.json()
    assert data_add["success"] is True
    assert data_add["added"] == 2

    resp_after = client.get("/model-info")
    assert resp_after.status_code == 200
    geo_points_after = resp_after.json()["geo_points"]
    assert geo_points_after == initial_geo_points + 2

@pytest.mark.integration
def test_high_price_observations_increase_prediction():
    house_payload = {
        "latitude": 46.5,
        "longitude": 6.5,
        "surface_m2": 50.0,
        "num_rooms": 2,
        "type": "room",
        "is_furnished": True,
        "floor": 1,
        "wifi_incl": True,
        "charges_incl": False,
        "car_park": True,
        "dist_public_transport_km": 0.5,
        "proxim_hesso_km": 1.2
    }

    resp_initial = client.post("/estimate-price", json=house_payload)
    assert resp_initial.status_code == 200
    initial_price = resp_initial.json()["predicted_price_chf"]

    high_price_obs = [
        {"latitude": 46.5, "longitude": 6.5, "price_chf": initial_price * 10},
        {"latitude": 46.5005, "longitude": 6.5005, "price_chf": initial_price * 8}
    ]
    resp_add = client.post("/observations", json=high_price_obs, headers={"API_KEY": os.getenv('API_KEY')})
    assert resp_add.status_code == 200
    assert resp_add.json()["added"] == 2

    resp_after = client.post("/estimate-price", json=house_payload)
    assert resp_after.status_code == 200
    new_price = resp_after.json()["predicted_price_chf"]

    assert new_price >= initial_price, (
        f"Expected new price ({new_price}) >= initial price ({initial_price})"
    )

@pytest.mark.unit
def test_estimate_price_invalid_negative_values():
    payload = {
        "latitude": 46.5,
        "longitude": 6.5,
        "surface_m2": -50.0,       # invalide
        "num_rooms": -2,            # invalide
        "type": "room",
        "is_furnished": True,
        "floor": -1,                # invalide
        "wifi_incl": True,
        "charges_incl": False,
        "car_park": True,
        "dist_public_transport_km": -0.5,  # invalide
        "proxim_hesso_km": -1.2            # invalide
    }

    resp = client.post("/estimate-price", json=payload)
    assert resp.status_code == 422  # erreur de validation Pydantic
    data = resp.json()
    assert "detail" in data

    # Vérifier que toutes les erreurs sont bien mentionnées
    error_fields = [e["loc"][-1] for e in data["detail"]]
    for field in ["surface_m2", "num_rooms", "floor", "dist_public_transport_km", "proxim_hesso_km"]:
        assert field in error_fields

@pytest.mark.unit
def test_estimate_price_missing_or_wrong_fields():
    payload = {
        "latitude": "not_a_float",
        "surface_m2": 50.0,
        "num_rooms": 2,
        "type": "invalid_type",
        "is_furnished": True,
        "floor": 1,
        "wifi_incl": True,
        "charges_incl": False,
        "car_park": True,
        "dist_public_transport_km": 0.5,
    }

    resp = client.post("/estimate-price", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data

    expected_error_fields = ["latitude", "longitude", "type", "proxim_hesso_km"]
    error_fields = [e["loc"][-1] for e in data["detail"]]
    for field in expected_error_fields:
        assert field in error_fields

@pytest.mark.unit
def test_add_observations_empty_payload():
    empty_payload = []

    resp = client.post("/observations", json=empty_payload, headers={"API_KEY": os.getenv('API_KEY')})
    assert resp.status_code == 400
    data = resp.json()
    assert "Empty payload." in data['detail']

@pytest.mark.integration
def test_add_observations_without_token():
    payload = [
        {"latitude": 46.5, "longitude": 6.5, "price_chf": 1500},
        {"latitude": 46.6, "longitude": 6.6, "price_chf": 1600}
    ]
    resp = client.post("/observations", json=payload)
    assert resp.status_code == 401
