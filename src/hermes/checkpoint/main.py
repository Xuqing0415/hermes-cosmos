"""
Hermes Checkpoint Service - Distributed checkpoint management for AI training
"""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from hermes.core.config import CheckpointConfig
from hermes.checkpoint.api import router
from hermes.checkpoint.storage.memory import MemoryStorage
from hermes.checkpoint.storage.pmem import PMemStorage
from hermes.checkpoint.storage.s3 import S3Storage
from hermes.checkpoint.delta import DeltaEngine
from hermes.checkpoint.coordinator import CheckpointCoordinator

logger = structlog.get_logger()

CHECKPOINT_COUNT = Counter(
    "hermes_checkpoint_total",
    "Total number of checkpoints",
    ["type", "status"],
)

CHECKPOINT_SIZE = Histogram(
    "hermes_checkpoint_size_bytes",
    "Checkpoint size in bytes",
    ["type"],
)

CHECKPOINT_LATENCY = Histogram(
    "hermes_checkpoint_latency_seconds",
    "Checkpoint creation latency in seconds",
    ["type"],
)


class CheckpointService:
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.app: FastAPI | None = None
        self.storage = None
        self.delta_engine: DeltaEngine | None = None
        self.coordinator: CheckpointCoordinator | None = None
        self._shutdown_event = asyncio.Event()

    async def setup(self) -> None:
        logger.info("Setting up Hermes Checkpoint Service")

        self.storage = await self._init_storage()

        self.delta_engine = DeltaEngine(
            enabled=self.config.storage.delta_enabled,
            compression=self.config.storage.compression,
        )

        self.coordinator = CheckpointCoordinator(
            storage=self.storage,
            delta_engine=self.delta_engine,
        )

        self._setup_telemetry()

        logger.info("Checkpoint service setup complete")

    async def _init_storage(self):
        backend = self.config.storage.backend

        if backend == "memory":
            return MemoryStorage()
        elif backend == "pmem":
            return PMemStorage(path=self.config.storage.pmem_path)
        elif backend == "s3":
            return S3Storage(
                bucket=self.config.storage.s3_bucket,
                region=self.config.storage.s3_region,
            )
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    async def teardown(self) -> None:
        logger.info("Tearing down Hermes Checkpoint Service")

        if self.storage:
            await self.storage.close()

        logger.info("Checkpoint service teardown complete")

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
            title="Hermes Checkpoint Service",
            description="Distributed Checkpoint Management for AI Training",
            version="1.0.0",
            lifespan=lifespan,
        )

        self.app.include_router(router, prefix="/v1/checkpoints")

        @self.app.get("/metrics")
        async def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type="text/plain",
            )

        FastAPIInstrumentor.instrument_app(self.app)

        return self.app

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
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self._shutdown(server)),
                )
        else:
            # Windows: use signal.signal instead of add_signal_handler
            def _win_shutdown():
                asyncio.ensure_future(self._shutdown(server), loop=loop)
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, lambda s, f: _win_shutdown())
                except (ValueError, OSError):
                    pass

        await server.serve()

    async def _shutdown(self, server: uvicorn.Server) -> None:
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
        await server.shutdown()


def main() -> None:
    config = CheckpointConfig()
    service = CheckpointService(config)
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
