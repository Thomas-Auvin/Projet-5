# db/models.py
from __future__ import annotations
from datetime import datetime
import uuid

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Float, Integer
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    pass

# JSON portable : JSONB en Postgres, JSON sinon (SQLite pour la CI)


def JSONPortable():
    return JSON().with_variant(JSONB, "postgresql")


class PredInput(Base):
    __tablename__ = "pred_inputs"

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True
        )
    uid: Mapped[str] = mapped_column(
        String(36), index=True, default=lambda: str(uuid.uuid4())
        )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
        )
    model_version: Mapped[str] = mapped_column(String(50))
    threshold: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(
        JSONPortable()
        )  # features reçues


class PredOutput(Base):
    __tablename__ = "pred_outputs"

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True
        )
    input_uid: Mapped[str] = mapped_column(
        String(36), index=True
        )  # FK logique vers PredInput.uid
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
        )
    proba: Mapped[float] = mapped_column(Float)
    label: Mapped[int] = mapped_column(Integer)
    served_by: Mapped[str] = mapped_column(
        String(32), default="api"
        )  # "api", "batch", etc.
