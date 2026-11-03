"""End-to-end smoke tests for the assembled FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app


def test_application_exposes_health_route_and_openapi_schema() -> None:
    with TestClient(app) as client:
        health_response = client.get("/health")
        openapi_response = client.get("/openapi.json")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert openapi_response.status_code == 200
    assert "/health" in openapi_response.json()["paths"]
