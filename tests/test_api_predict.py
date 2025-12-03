# tests/test_api_predict.py
import io
import pandas as pd


def test_predict_one_ok(test_client_overriding):
    client = test_client_overriding
    payload = {"features": {"foo": 1, "bar": "x"}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["proba"] <= 1.0
    assert data["label"] in (0, 1)


def test_predict_batch_ok(test_client_overriding):
    client = test_client_overriding
    rows = [{"a": 1}, {"a": 2}]
    r = client.post("/predict_batch", json={"rows": rows})
    assert r.status_code == 200
    data = r.json()
    assert data["threshold"] is not None
    assert len(data["items"]) == 2
    assert all(0.0 <= it["proba"] <= 1.0 for it in data["items"])


def test_predict_batch_empty(test_client_overriding):
    client = test_client_overriding
    r = client.post("/predict_batch", json={"rows": []})
    assert r.status_code == 200
    assert r.json() == {"items": []}


def test_predict_csv_ok(test_client_overriding):
    client = test_client_overriding
    df = pd.DataFrame([{"x": 1}, {"x": 2}])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    files = {"file": ("mini.csv", buf, "text/csv")}
    r = client.post("/predict_csv", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "mini.csv"
    assert data["n_rows"] == 2
    assert len(data["items"]) == 2
