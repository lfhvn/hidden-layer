"""
Rate limiting and authentication utilities for web tools.
"""

import time
from collections import defaultdict
from functools import wraps
from typing import Dict, Optional
from fastapi import HTTPException, Request, Header
import hashlib


class RateLimiter:
    """Simple in-memory rate limiter based on IP address."""

    def __init__(self, requests: int = 5, window: int = 3600):
        """
        Args:
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.storage: Dict[str, list] = defaultdict(list)

    def _get_key(self, request: Request) -> str:
        """Get identifier for rate limiting (IP address)."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return hashlib.md5(ip.encode()).hexdigest()[:16]

    def _clean_old_requests(self, key: str, now: float):
        """Remove requests outside the time window."""
        cutoff = now - self.window
        self.storage[key] = [ts for ts in self.storage[key] if ts > cutoff]

    def check(self, request: Request) -> bool:
        """Check if request is allowed. Raises HTTPException if rate limited."""
        key = self._get_key(request)
        now = time.time()

        # Clean old requests
        self._clean_old_requests(key, now)

        # Check limit
        if len(self.storage[key]) >= self.requests:
            oldest = self.storage[key][0]
            retry_after = int(oldest + self.window - now)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": self.requests,
                    "window": self.window,
                    "retry_after": retry_after
                }
            )

        # Record this request
        self.storage[key].append(now)
        return True

    def get_usage(self, request: Request) -> dict:
        """Get current usage for an IP."""
        key = self._get_key(request)
        now = time.time()
        self._clean_old_requests(key, now)

        used = len(self.storage[key])
        remaining = max(0, self.requests - used)

        if self.storage[key]:
            oldest = self.storage[key][0]
            reset_time = int(oldest + self.window)
        else:
            reset_time = int(now + self.window)

        return {
            "limit": self.requests,
            "used": used,
            "remaining": remaining,
            "reset": reset_time,
            "window": self.window
        }


class APIKeyValidator:
    """Validate API keys for bring-your-own-key mode."""

    def __init__(self, allow_byok: bool = True):
        self.allow_byok = allow_byok

    def validate_anthropic_key(self, api_key: str) -> bool:
        """Basic validation of Anthropic API key format."""
        if not api_key:
            return False
        return api_key.startswith("sk-ant-") and len(api_key) > 20

    def extract_key(
        self,
        request: Request,
        x_api_key: Optional[str] = Header(None),
        authorization: Optional[str] = Header(None)
    ) -> Optional[str]:
        """Extract API key from headers."""
        # Try X-API-Key header
        if x_api_key:
            return x_api_key

        # Try Authorization header
        if authorization:
            if authorization.startswith("Bearer "):
                return authorization[7:]
            return authorization

        return None


def create_rate_limited_endpoint(
    limiter: RateLimiter,
    api_key_validator: Optional[APIKeyValidator] = None
):
    """
    Decorator factory for rate-limited endpoints.

    Usage:
        limiter = RateLimiter(requests=5, window=3600)
        validator = APIKeyValidator()

        @app.post("/api/generate")
        @create_rate_limited_endpoint(limiter, validator)
        async def generate(request: Request, user_api_key: Optional[str] = None):
            # If user_api_key is provided, it's validated and unlimited
            # Otherwise, rate limiting applies
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user provided their own API key
            if api_key_validator:
                user_key = api_key_validator.extract_key(request)
                if user_key and api_key_validator.validate_anthropic_key(user_key):
                    # Valid user key - bypass rate limiting
                    kwargs["user_api_key"] = user_key
                    return await func(request, *args, **kwargs)

            # No valid user key - apply rate limiting
            limiter.check(request)
            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


# Default instances
default_limiter = RateLimiter(requests=5, window=3600)
default_api_key_validator = APIKeyValidator()
