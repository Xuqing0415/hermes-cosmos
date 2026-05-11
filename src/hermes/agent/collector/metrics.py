"""
System metrics collector
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class MetricSample:
    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str]


class MetricsCollector:
    def __init__(self, interval_seconds: int = 10) -> None:
        self.interval_seconds = interval_seconds
        self._last_collection: datetime | None = None

    async def collect(self) -> dict[str, Any]:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": await self._collect_cpu_metrics(),
            "memory": await self._collect_memory_metrics(),
            "disk": await self._collect_disk_metrics(),
            "network": await self._collect_network_metrics(),
        }

        self._last_collection = datetime.utcnow()
        return metrics

    async def _collect_cpu_metrics(self) -> dict[str, Any]:
        return {
            "utilization": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "load_avg": list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else [0, 0, 0],
        }

    async def _collect_memory_metrics(self) -> dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
        }

    async def _collect_disk_metrics(self) -> dict[str, Any]:
        disk = psutil.disk_usage("/")
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        }

    async def _collect_network_metrics(self) -> dict[str, Any]:
        net = psutil.net_io_counters()
        return {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        }
