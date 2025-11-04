"""Database connection and session management."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool

from ..config import get_settings

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None


def get_engine():
    """Get or create database engine."""
    global _engine

    if _engine is None:
        settings = get_settings()
        database_url = settings.database_url

        # Special handling for SQLite
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            # Use StaticPool for in-memory databases
            if ":memory:" in database_url:
                _engine = create_engine(
                    database_url,
                    connect_args=connect_args,
                    poolclass=StaticPool,
                )
            else:
                _engine = create_engine(
                    database_url,
                    connect_args=connect_args,
                )
        else:
            # PostgreSQL or other databases
            _engine = create_engine(database_url, pool_pre_ping=True)

        logger.info(f"Created database engine for {database_url}")

    return _engine


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session context manager.

    Usage:
        with get_session() as session:
            session.add(obj)
            session.commit()
    """
    engine = get_engine()
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
