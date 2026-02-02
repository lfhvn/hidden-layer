"""
Database session management for AgentMesh.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agentmesh.db.models import Base


# Database URL (will be configurable via env)
DATABASE_URL = "postgresql+asyncpg://agentmesh:agentmesh@localhost:5432/agentmesh"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL queries (disable in production)
    future=True,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Initialize database (create tables)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session (async context manager)"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
