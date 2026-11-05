"""End-to-end smoke tests for the assembled FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app, create_app
from app.repositories.models import AvailableAppointmentSlot


class _EmptyAppointmentService:
    def search_available_slots(
        self,
        starts_at,
        ends_at,
        *,
        doctor_id=None,
        limit=25,
    ) -> list[AvailableAppointmentSlot]:
        return []


class _UnusedConversationService:
    pass


def test_application_exposes_health_route_and_openapi_schema() -> None:
    with TestClient(app) as client:
        health_response = client.get("/health")
        openapi_response = client.get("/openapi.json")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert openapi_response.status_code == 200
    assert "/health" in openapi_response.json()["paths"]
    assert "/appointments/slots" in openapi_response.json()["paths"]
    assert "/approvals/pending" in openapi_response.json()["paths"]


def test_appointment_router_and_service_are_wired_in_app_factory() -> None:
    application = create_app(
        appointment_service=_EmptyAppointmentService(),
        conversation_service=_UnusedConversationService(),
        configure_services=False,
    )

    with TestClient(application) as client:
        response = client.get(
            "/appointments/slots",
            params={
                "starts_at": "2026-12-01T09:00:00Z",
                "ends_at": "2026-12-01T10:00:00Z",
            },
        )

    assert response.status_code == 200
    assert response.json() == []
