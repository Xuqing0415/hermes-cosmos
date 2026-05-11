"""
Rate limiting middleware
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from hermes.core.exceptions import RateLimitError

logger = structlog.get_logger()


class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate,
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def time_until_available(self, tokens: int = 1) -> float:
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        rps: int = 100,
        burst: int = 200,
        per_tenant: bool = True,
        excluded_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.rps = rps
        self.burst = burst
        self.per_tenant = per_tenant
        self.excluded_paths = excluded_paths or {
            "/v1/health",
            "/v1/ready",
            "/v1/live",
            "/metrics",
        }
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(rps, burst)
        )
        self._global_bucket = TokenBucket(rps * 10, burst * 10)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        key = self._get_rate_limit_key(request)

        if not self._check_rate_limit(key):
            retry_after = self._get_retry_after(key)
            logger.warning(
                "Rate limit exceeded",
                key=key,
                retry_after=retry_after,
            )
            raise RateLimitError(retry_after=int(retry_after))

        return await call_next(request)

    def _get_rate_limit_key(self, request: Request) -> str:
        if self.per_tenant:
            tenant_id = getattr(request.state, "tenant_id", None)
            if tenant_id:
                return f"tenant:{tenant_id}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _check_rate_limit(self, key: str) -> bool:
        if not self._global_bucket.consume():
            return False

        with self._lock:
            bucket = self._buckets[key]
            return bucket.consume()

    def _get_retry_after(self, key: str) -> float:
        with self._lock:
            bucket = self._buckets[key]
            return bucket.time_until_available()
