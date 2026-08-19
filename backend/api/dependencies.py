"""FastAPI Dependency Injection providers."""

from typing import Generator
from sqlalchemy.orm import Session
from backend.models.db import get_session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session per request."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
