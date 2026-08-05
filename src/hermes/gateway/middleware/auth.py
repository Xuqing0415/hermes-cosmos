"""
Authentication middleware using SPIFFE
"""

import time
from typing import Callable, Optional

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from hermes.core.exceptions import AuthenticationError, AuthorizationError

logger = structlog.get_logger()


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        spiffe_host: Optional[str] = None,
        jwks_url: Optional[str] = None,
        excluded_paths: Optional[set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.spiffe_host = spiffe_host
        self.jwks_url = jwks_url
        self.excluded_paths = excluded_paths or {
            "/v1/health",
            "/v1/ready",
            "/v1/live",
            "/metrics",
            "/docs",
            "/openapi.json",
        }
        self._jwks_cache: dict = {}
        self._jwks_cache_time: float = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"error": "AUTHENTICATION_ERROR", "message": "Missing Authorization header"},
            )

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={"error": "AUTHENTICATION_ERROR", "message": "Invalid Authorization header format"},
            )

        token = parts[1]

        try:
            claims = await self._validate_token(token)
            request.state.user = claims
        except AuthenticationError as e:
            return JSONResponse(
                status_code=401,
                content={"error": e.code, "message": e.message},
            )

        return await call_next(request)

    async def _validate_token(self, token: str) -> dict:
        try:
            unverified = jwt.decode(token, key="", options={"verify_signature": False})
            return unverified
        except JWTError:
            raise AuthenticationError("Invalid token format")

    async def _get_jwks(self) -> dict:
        if not self.jwks_url:
            raise AuthenticationError("JWKS URL not configured")

        current_time = time.time()
        if self._jwks_cache and current_time - self._jwks_cache_time < 3600:
            return self._jwks_cache

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_time = current_time
            return self._jwks_cache
