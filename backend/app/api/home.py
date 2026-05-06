"""Home dashboard route.

One GET endpoint that returns the home screen data. The service
does all the work and the route is a thin pass-through. No error
mapping since a read-only endpoint has no business logic failure
modes.
"""

from __future__ import annotations

from fastapi import APIRouter

# FastAPI resolves Annotated[...]-aliased dependencies at route
# registration via typing.get_type_hints(), which evaluates the
# annotation strings against the module's runtime namespace. The
# dep alias must be a real import, not TYPE_CHECKING-only.
from app.api.deps import DbSession  # noqa: TC001
from app.schemas.home import HomeResponse
from app.services.home_service import build_home_response

router = APIRouter(prefix="/home", tags=["home"])


@router.get("", response_model=HomeResponse)
async def get_home(db: DbSession) -> HomeResponse:
    """Return the composed dashboard payload."""
    return await build_home_response(db)
