"""FastAPI application entry point for the clinic front-desk assistant."""

from fastapi import APIRouter, FastAPI

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is ready to serve requests."""
    return {"status": "ok"}


app = FastAPI(
    title="Clinic Front-Desk Assistant API",
    version="0.1.0",
)
app.include_router(api_router)
