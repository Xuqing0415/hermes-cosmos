"""
Hermes Agent - Local cluster agent for fault prediction and self-healing
"""

import asyncio
import signal
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
from prometheus_client import Counter, Gauge, generate_latest
from starlette.responses import Response

from hermes.core.config import AgentConfig
from hermes.agent.api import router
from hermes.agent.collector.metrics import MetricsCollector
from hermes.agent.collector.gpu import GPUMetricsCollector
from hermes.agent.prediction.fault_predictor import FaultPredictor
from hermes.agent.healing.self_healing import SelfHealingEngine

logger = structlog.get_logger()

PREDICTIONS_TOTAL = Counter(
    "hermes_agent_predictions_total",
    "Total fault predictions",
    ["type", "severity"],
)

GPU_UTILIZATION = Gauge(
    "hermes_agent_gpu_utilization",
    "GPU utilization percentage",
    ["gpu_id"],
)

GPU_MEMORY_USED = Gauge(
    "hermes_agent_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
)


class HermesAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.app: FastAPI | None = None
        self.metrics_collector: MetricsCollector | None = None
        self.gpu_collector: GPUMetricsCollector | None = None
        self.fault_predictor: FaultPredictor | None = None
        self.healing_engine: SelfHealingEngine | None = None
        self._shutdown_event = asyncio.Event()
        self._collection_task: asyncio.Task | None = None
        self._prediction_task: asyncio.Task | None = None

    async def setup(self) -> None:
        logger.info("Setting up Hermes Agent", region=self.config.region.value)

        self.metrics_collector = MetricsCollector(
            interval_seconds=10,
        )

        self.gpu_collector = GPUMetricsCollector(
            interval_seconds=10,
        )

        self.fault_predictor = FaultPredictor(
            model_path=self.config.fault_prediction.model_path,
            confidence_threshold=self.config.fault_prediction.confidence_threshold,
            prediction_window=self.config.fault_prediction.prediction_window,
        )

        if self.config.fault_prediction.enabled:
            await self.fault_predictor.load_model()

        self.healing_engine = SelfHealingEngine(
            agent=self,
        )

        self._setup_telemetry()

        self._collection_task = asyncio.create_task(self._run_collection_loop())
        self._prediction_task = asyncio.create_task(self._run_prediction_loop())

        logger.info("Agent setup complete")

    async def teardown(self) -> None:
        logger.info("Tearing down Hermes Agent")

        self._shutdown_event.set()

        if self._collection_task:
            self._collection_task.cancel()
        if self._prediction_task:
            self._prediction_task.cancel()

        logger.info("Agent teardown complete")

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

    async def _run_collection_loop(self) -> None:
        logger.info("Starting metrics collection loop")

        while not self._shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(5)

        logger.info("Metrics collection loop stopped")

    async def _collect_metrics(self) -> None:
        assert self.gpu_collector is not None

        gpu_metrics = await self.gpu_collector.collect()

        for gpu_id, metrics in gpu_metrics.items():
            GPU_UTILIZATION.labels(gpu_id=gpu_id).set(metrics.get("utilization", 0))
            GPU_MEMORY_USED.labels(gpu_id=gpu_id).set(metrics.get("memory_used", 0))

    async def _run_prediction_loop(self) -> None:
        if not self.config.fault_prediction.enabled:
            return

        logger.info("Starting fault prediction loop")

        while not self._shutdown_event.is_set():
            try:
                await self._run_prediction()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Fault prediction error", error=str(e))
                await asyncio.sleep(30)

        logger.info("Fault prediction loop stopped")

    async def _run_prediction(self) -> None:
        assert self.fault_predictor is not None
        assert self.metrics_collector is not None

        metrics = await self.metrics_collector.collect()

        predictions = await self.fault_predictor.predict(metrics)

        for prediction in predictions:
            PREDICTIONS_TOTAL.labels(
                type=prediction.prediction_type,
                severity="high" if prediction.probability > 0.8 else "medium",
            ).inc()

            logger.warning(
                "Fault prediction",
                prediction_type=prediction.prediction_type,
                probability=prediction.probability,
                predicted_time=prediction.predicted_failure_time.isoformat(),
                recommended_action=prediction.recommended_action,
            )

            if prediction.probability >= self.config.fault_prediction.confidence_threshold:
                await self.healing_engine.handle_prediction(prediction)

    def create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            await self.setup()
            yield
            await self.teardown()

        self.app = FastAPI(
            title="Hermes Agent",
            description="Local Cluster Agent for Fault Prediction and Self-Healing",
            version="1.0.0",
            lifespan=lifespan,
        )

        self.app.include_router(router, prefix="/v1")

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
    config = AgentConfig()
    agent = HermesAgent(config)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
