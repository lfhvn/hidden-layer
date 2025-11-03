"""API dependencies and middleware."""

from fastapi import Header, HTTPException, status

from ..config import get_settings


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is invalid
    """
    settings = get_settings()

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key
