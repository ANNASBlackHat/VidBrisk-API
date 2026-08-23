"""Database engine and session management for backend service."""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator
from sqlalchemy import create_engine, inspect, text
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

        try:
            _engines[url] = create_engine(url, connect_args=connect_args, **engine_kwargs)
        except Exception:
            sqlite_url = "sqlite:///data/jobs.db"
            if sqlite_url not in _engines:
                _engines[sqlite_url] = create_engine(sqlite_url, connect_args={"check_same_thread": False})
            return _engines[sqlite_url]
    return _engines[url]


def _migrate_columns(engine) -> None:
    """Ensures newly added model columns exist in existing database tables."""
    try:
        inspector = inspect(engine)
        if "video_jobs" in inspector.get_table_names():
            columns = {col["name"] for col in inspector.get_columns("video_jobs")}
            with engine.begin() as conn:
                if "motion_qa_thumbnails" not in columns:
                    if engine.dialect.name == "postgresql":
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS motion_qa_thumbnails JSON;"))
                    else:
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN motion_qa_thumbnails JSON;"))
                if "progress" not in columns:
                    if engine.dialect.name == "postgresql":
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS progress JSON;"))
                    else:
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN progress JSON;"))
                if "title" not in columns:
                    if engine.dialect.name == "postgresql":
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS title VARCHAR(255);"))
                    else:
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN title VARCHAR(255);"))
                if "video_url" not in columns:
                    if engine.dialect.name == "postgresql":
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN IF NOT EXISTS video_url VARCHAR(500);"))
                    else:
                        conn.execute(text("ALTER TABLE video_jobs ADD COLUMN video_url VARCHAR(500);"))
    except Exception as e:
        # Non-fatal if table not created yet or permission restricted
        pass


def init_db(database_url: str | None = None) -> None:
    """Creates all database tables defined on Base and applies safe lightweight migrations."""
    try:
        engine = get_engine(database_url)
        Base.metadata.create_all(bind=engine)
        _migrate_columns(engine)
    except Exception:
        fallback_engine = get_engine("sqlite:///data/jobs.db")
        Base.metadata.create_all(bind=fallback_engine)
        _migrate_columns(fallback_engine)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or get_settings().DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    if url not in _session_factories:
        engine = get_engine(url)
        _migrate_columns(engine)
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
