# tests/test_api_smoke.py
# tests/test_api_smoke.py


def test_root_redirect(test_client_overriding):
    client = test_client_overriding
    r = client.get("/", follow_redirects=False)

    assert r.status_code in (301, 302, 307, 308)
    assert r.headers.get("location") in ("/docs", "/docs/")


def test_health_ok(test_client_overriding):
    client = test_client_overriding
    r = client.get("/health")

    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "version" in data


def test_meta_ok(test_client_overriding):
    client = test_client_overriding
    r = client.get("/meta")

    assert r.status_code == 200
    data = r.json()

    assert "version" in data
    assert "threshold" in data
    assert "model_meta" in data
