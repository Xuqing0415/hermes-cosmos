"""
Hermes Gateway - Unified API Gateway
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from hermes.core.config import GatewayConfig
from hermes.core.exceptions import HermesError, RateLimitError
from hermes.gateway.api import jobs, checkpoints, resources, health
from hermes.gateway.middleware.auth import AuthMiddleware
from hermes.gateway.middleware.logging import LoggingMiddleware
from hermes.gateway.middleware.rate_limit import RateLimitMiddleware
from hermes.gateway.services.scheduler import SchedulerClient
from hermes.gateway.services.checkpoint import CheckpointClient

logger = structlog.get_logger()

REQUEST_COUNT = Counter(
    "hermes_gateway_requests_total",
    "Total request count",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "hermes_gateway_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)


class Gateway:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.app: FastAPI | None = None
        self.scheduler_client: SchedulerClient | None = None
        self.checkpoint_client: CheckpointClient | None = None
        self._shutdown_event = asyncio.Event()

    async def setup(self) -> None:
        logger.info("Setting up Hermes Gateway")

        self.scheduler_client = SchedulerClient(self.config.upstream.scheduler_addr)
        self.checkpoint_client = CheckpointClient(self.config.upstream.checkpoint_addr)

        await self.scheduler_client.connect()
        await self.checkpoint_client.connect()

        self._setup_telemetry()

        logger.info("Gateway setup complete")

    async def teardown(self) -> None:
        logger.info("Tearing down Hermes Gateway")

        if self.scheduler_client:
            await self.scheduler_client.disconnect()
        if self.checkpoint_client:
            await self.checkpoint_client.disconnect()

        logger.info("Gateway teardown complete")

    def _setup_telemetry(self) -> None:
        if not self.config.tracing.enabled:
            return

        resource = Resource.create({"service.name": self.config.tracing.service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=self.config.tracing.endpoint)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    def create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            await self.setup()
            yield
            await self.teardown()

        self.app = FastAPI(
            title="Hermes Gateway",
            description="Unified API Gateway for Hermes Global AI Training System",
            version="1.0.0",
            lifespan=lifespan,
        )

        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()

        FastAPIInstrumentor.instrument_app(self.app)

        return self.app

    def _setup_middleware(self) -> None:
        assert self.app is not None

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.add_middleware(LoggingMiddleware)

        if self.config.rate_limit.enabled:
            self.app.add_middleware(
                RateLimitMiddleware,
                rps=self.config.rate_limit.rps,
                burst=self.config.rate_limit.burst,
                per_tenant=self.config.rate_limit.per_tenant,
            )

        if self.config.auth.enabled:
            self.app.add_middleware(
                AuthMiddleware,
                spiffe_host=self.config.auth.spiffe_host,
                jwks_url=self.config.auth.jwks_url,
            )

    def _setup_routes(self) -> None:
        assert self.app is not None

        self.app.include_router(jobs.router, prefix="/v1/jobs", tags=["Jobs"])
        self.app.include_router(checkpoints.router, prefix="/v1/checkpoints", tags=["Checkpoints"])
        self.app.include_router(resources.router, prefix="/v1/resources", tags=["Resources"])
        self.app.include_router(health.router, prefix="/v1", tags=["Health"])

        @self.app.get("/metrics")
        async def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type="text/plain",
            )

    def _setup_exception_handlers(self) -> None:
        assert self.app is not None

        @self.app.exception_handler(HermesError)
        async def hermes_error_handler(request: Request, exc: HermesError) -> JSONResponse:
            logger.error(
                "Hermes error",
                error=exc.code,
                message=exc.message,
                details=exc.details,
            )
            status_code = 500
            if isinstance(exc, RateLimitError):
                status_code = 429
            return JSONResponse(
                status_code=status_code,
                content=exc.to_dict(),
            )

        @self.app.exception_handler(Exception)
        async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
            logger.exception("Unexpected error", error=str(exc))
            return JSONResponse(
                status_code=500,
                content={"error": "INTERNAL_ERROR", "message": "Internal server error"},
            )

    async def run(self) -> None:
        app = self.create_app()

        config = uvicorn.Config(
            app,
            host=self.config.server.host,
            port=self.config.server.port,
            workers=self.config.server.workers,
            log_level=self.config.log_level.value.lower(),
            access_log=False,
        )

        server = uvicorn.Server(config)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._shutdown(server)),
            )

        await server.serve()

    async def _shutdown(self, server: uvicorn.Server) -> None:
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
        await server.shutdown()


def main() -> None:
    config = GatewayConfig()
    gateway = Gateway(config)
    asyncio.run(gateway.run())


if __name__ == "__main__":
    main()
