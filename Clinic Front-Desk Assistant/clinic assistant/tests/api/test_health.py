"""API tests for the FastAPI health endpoint."""

from fastapi import status
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint_returns_ok() -> None:
    app = create_app(
        appointment_service=object(),
        conversation_service=object(),
        approval_service=object(),
        owner_reference_resolver=lambda request: "opaque-test-owner",
        staff_reference_resolver=lambda request: "opaque-test-staff",
        configure_services=False,
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}
