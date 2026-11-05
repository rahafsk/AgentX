"""Shared HTTP dependencies for trusted caller context."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status


def get_authenticated_owner_reference(request: Request) -> str:
    """Return the opaque owner ID attached by trusted authentication middleware."""
    owner_reference = getattr(request.state, "owner_reference", None)
    if not isinstance(owner_reference, str) or not owner_reference.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return owner_reference.strip()


def get_authenticated_staff_reference(request: Request) -> str:
    """Return the staff ID attached by trusted staff-authentication middleware."""
    staff_reference = getattr(request.state, "staff_reference", None)
    if not isinstance(staff_reference, str) or not staff_reference.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Staff authentication required.",
        )
    return staff_reference.strip()


OwnerReferenceDependency = Annotated[
    str,
    Depends(get_authenticated_owner_reference),
]
StaffReferenceDependency = Annotated[
    str,
    Depends(get_authenticated_staff_reference),
]
IdempotencyKey = Annotated[UUID, Header(alias="Idempotency-Key")]
