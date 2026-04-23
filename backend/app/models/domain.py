"""Top-level domains for the topic tree.

A `Domain` is the top of a topic path. For example, in the
path `Python > Data Types > Integers`, the domain is `Python`.
Every topic belongs to one domain.

The `name` field is unique, so code can use it as a stable ID.
A fresh install fills this table with a known set of domains.
The `kind` field says what sort of thing the domain is: a
language, a framework, a concept, a tool, and so on.
"""

from __future__ import annotations

from sqlalchemy import Enum as SQLEnum
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import DomainKind


class Domain(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "domain"

    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    kind: Mapped[DomainKind] = mapped_column(
        SQLEnum(DomainKind, native_enum=False, length=32),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<Domain name={self.name!r} kind={self.kind.value!r}>"
