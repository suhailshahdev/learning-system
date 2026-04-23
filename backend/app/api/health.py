"""Health check endpoint.

Reports whether the process is alive and whether its key
dependencies are reachable. The response body is structured so
each component can be checked on its own. Callers that only care
about a plain up/down signal can read the top-level status.
"""

from typing import Literal

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import DbSession

router = APIRouter(prefix="/health", tags=["health"])

HealthStatus = Literal["ok", "degraded"]
ComponentStatus = Literal["ok", "error"]


class ComponentHealth(BaseModel):
    """Status of a single dependency"""

    status: ComponentStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    """Top-level health payload

    `status` is `ok` when every component is `ok`, otherwise
    `degraded`. Clients that want a simple up/down answer should
    read `status`. Clients that want to show which thing broke
    should read `components`.
    """

    status: HealthStatus
    components: dict[str, ComponentHealth]


@router.get("", response_model=HealthResponse)
def health(db: DbSession) -> JSONResponse:
    """Return the service's health

    Runs a tiny query against the database to confirm the engine
    can open a connection. This is the check worth having from day
    one, because a broken database connection is the most common
    thing that goes wrong in local development (file missing,
    file locked, schema drifted from models).
    """
    components: dict[str, ComponentHealth] = {}

    try:
        db.execute(text("SELECT 1"))
        components["database"] = ComponentHealth(status="ok")
    except SQLAlchemyError as exc:
        components["database"] = ComponentHealth(status="error", detail=str(exc))

    overall: HealthStatus = (
        "ok" if all(c.status == "ok" for c in components.values()) else "degraded"
    )
    payload = HealthResponse(status=overall, components=components)

    http_status = status.HTTP_200_OK if overall == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=payload.model_dump())
