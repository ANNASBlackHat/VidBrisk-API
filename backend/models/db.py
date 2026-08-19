"""Database engine and session management for backend service."""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from pipeline.config import get_settings


def utc_now() -> datetime:
    """Returns timezone-aware UTC current time."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """SQLAlchemy Declarative Base."""
    pass


_engines: dict[str, Any] = {}
_session_factories: dict[str, sessionmaker[Session]] = {}


def get_engine(database_url: str | None = None):
    url = database_url or get_settings().DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    if url not in _engines:
        connect_args = {}
        engine_kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 5
            engine_kwargs["pool_recycle"] = 120
            connect_args = {"connect_timeout": 10}
        _engines[url] = create_engine(url, connect_args=connect_args, **engine_kwargs)
    return _engines[url]


def init_db(database_url: str | None = None) -> None:
    """Creates all database tables defined on Base."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or get_settings().DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    if url not in _session_factories:
        engine = get_engine(url)
        _session_factories[url] = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            bind=engine,
        )
    return _session_factories[url]


@contextmanager
def get_db_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """Context manager for safe transactional database sessions."""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
