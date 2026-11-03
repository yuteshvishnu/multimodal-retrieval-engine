"""The optional app password (APP_PASSWORD)."""
import base64

from fastapi.testclient import TestClient

import backend.main as main


def basic(password):
    return {"Authorization": "Basic " + base64.b64encode(f"me:{password}".encode()).decode()}


def test_no_password_set_means_open(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    assert TestClient(main.app).get("/api/health").status_code == 200


def test_password_required_when_set(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-only-password")
    client = TestClient(main.app)
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers=basic("wrong")).status_code == 401
    assert client.get("/api/health", headers=basic("test-only-password")).status_code == 200
