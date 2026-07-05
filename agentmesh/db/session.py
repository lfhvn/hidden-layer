"""
Database session management for AgentMesh.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agentmesh.db.models import Base


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://agentmesh:agentmesh@localhost:5432/agentmesh",
)

# Create async engine (set AGENTMESH_SQL_ECHO=1 to log SQL queries)
engine = create_async_engine(
    DATABASE_URL,
    echo=os.environ.get("AGENTMESH_SQL_ECHO", "").lower() in ("1", "true", "yes"),
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
