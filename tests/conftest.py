# tests/conftest.py
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_model


class FakeModel:
    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        p1 = np.full((n, 1), 0.6, dtype=float)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])


@pytest.fixture(autouse=True)
def test_client_overriding():
    app.dependency_overrides[get_model] = lambda: FakeModel()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
