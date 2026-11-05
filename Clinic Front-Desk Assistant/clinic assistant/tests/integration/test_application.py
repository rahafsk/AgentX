"""End-to-end smoke tests for the assembled FastAPI application."""

from fastapi.testclient import TestClient

from app.main import create_app
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


class _UnusedApprovalService:
    pass


def test_application_exposes_health_route_and_openapi_schema() -> None:
    application = create_app(configure_services=False)
    with TestClient(application) as client:
        health_response = client.get("/health")
        openapi_response = client.get("/openapi.json")

    assert health_response.status_code == 503
    assert health_response.json() == {"detail": "Application services are not ready."}
    assert openapi_response.status_code == 200
    assert "/health" in openapi_response.json()["paths"]
    assert "/appointments/slots" in openapi_response.json()["paths"]
    assert "/approvals/pending" in openapi_response.json()["paths"]


def test_appointment_router_and_service_are_wired_in_app_factory() -> None:
    application = create_app(
        appointment_service=_EmptyAppointmentService(),
        conversation_service=_UnusedConversationService(),
        approval_service=_UnusedApprovalService(),
        owner_reference_resolver=lambda request: "opaque-test-owner",
        staff_reference_resolver=lambda request: "opaque-test-staff",
        configure_services=False,
    )

    with TestClient(application) as client:
        health = client.get("/health")
        response = client.get(
            "/appointments/slots",
            params={
                "starts_at": "2026-12-01T09:00:00Z",
                "ends_at": "2026-12-01T10:00:00Z",
            },
        )

    assert response.status_code == 200
    assert response.json() == []
    assert health.status_code == 200
