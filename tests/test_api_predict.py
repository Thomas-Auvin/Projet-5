# tests/test_api_predict.py
from fastapi.testclient import TestClient
from app.main import app, get_model

class FakeModel:
    def predict_proba(self, X):
        import numpy as np
        # renvoie 0.8 pour toute ligne -> label=1 si seuil <= 0.8
        probs = np.full((len(X), 2), [0.2, 0.8])
        return probs

def override_get_model():
    return FakeModel()

client = TestClient(app)

def setup_module(_module=None):
    # on remplace la dépendance de chargement du modèle
    app.dependency_overrides[get_model] = override_get_model

def teardown_module(_module=None):
    app.dependency_overrides.clear()

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_one_ok():
    payload = {"features": {"age": 35, "salaire": 28000, "departement": "sales"}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "proba" in data and "label" in data
    assert 0 <= data["proba"] <= 1
    # Avec faux modèle proba=0.8 => label 1 si seuil <= 0.8
    assert data["label"] in (0, 1)

def test_predict_batch_ok():
    payload = {
        "rows": [
            {"age": 35, "salaire": 28000},
            {"age": 45, "salaire": 35000},
        ]
    }
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert all(0 <= it["proba"] <= 1 for it in data["items"])
