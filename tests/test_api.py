from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_meta_lists_countries():
    meta = client.get("/api/meta").json()
    assert len(meta["countries"]) == 101
    assert "Wheat" in meta["countries"]["India"]


def test_predict_has_ordered_interval():
    r = client.post("/api/predict", json={"area": "India", "item": "Wheat"}).json()
    assert r["low"] < r["yield_hg_ha"] < r["high"]
    assert r["year"] == 2014
    assert {d["name"] for d in r["drivers"]} >= {"Temperature", "Rainfall"}


def test_scenario_inputs_are_used():
    r = client.post("/api/predict", json={"area": "India", "item": "Wheat", "temp": 30}).json()
    assert r["inputs"]["temp"] == 30


def test_compare_covers_all_country_crops():
    rows = client.post("/api/compare", json={"area": "India"}).json()
    assert len(rows) == 8
    assert rows == sorted(rows, key=lambda x: -x["change_pct"])


def test_sensitivity_curve():
    pts = client.post("/api/sensitivity?variable=temp", json={"area": "Brazil", "item": "Maize"}).json()
    assert len(pts) == 25


def test_unknown_country_is_404():
    assert client.post("/api/predict", json={"area": "Atlantis", "item": "Wheat"}).status_code == 404


def test_missing_crop_is_422():
    assert client.post("/api/predict", json={"area": "India"}).status_code == 422
