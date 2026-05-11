"""
Request logging middleware
"""

import time
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        request_id = request.headers.get("X-Request-ID", "-")
        tenant_id = getattr(request.state, "tenant_id", "-")

        log = logger.bind(
            request_id=request_id,
            tenant_id=tenant_id,
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            client_ip=request.client.host if request.client else "-",
        )

        log.info("Request started")

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            log.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"

            return response

        except Exception as exc:
            duration = time.time() - start_time
            log.error(
                "Request failed",
                error=str(exc),
                duration_ms=round(duration * 1000, 2),
            )
            raise
