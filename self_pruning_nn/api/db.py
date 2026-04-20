from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Integer, Float, Text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class RagQueryLog(Base):
    __tablename__ = "rag_query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    question: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    filename: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    predicted_class: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)


def get_database_url() -> str:
    # default to local SQLite in api/ dir
    default_path = os.path.join(os.path.dirname(__file__), "app.db")
    path = os.getenv("SELF_PRUNING_DB_PATH", default_path)
    return f"sqlite+aiosqlite:///{path}"


def create_engine() -> AsyncEngine:
    return create_async_engine(get_database_url(), future=True)


def create_session_factory(engine: AsyncEngine) -> sessionmaker:
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

