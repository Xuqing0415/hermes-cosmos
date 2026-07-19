"""
GPU metrics collector
"""

from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


class GPUMetricsCollector:
    def __init__(self, interval_seconds: int = 10) -> None:
        self.interval_seconds = interval_seconds
        self._nvidia_available = False

    async def collect(self) -> dict[str, dict[str, Any]]:
        metrics = {}

        initialized = False
        try:
            import pynvml
            pynvml.nvmlInit()
            initialized = True
            self._nvidia_available = True

            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)

                gpu_id = f"gpu-{i}"
                metrics[gpu_id] = {
                    "utilization": util.gpu,
                    "memory_utilization": util.memory,
                    "memory_used": mem.used,
                    "memory_total": mem.total,
                    "memory_free": mem.free,
                    "temperature": temp,
                    "power_draw": power / 1000.0,
                    "power_limit": power_limit / 1000.0,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except ImportError:
            logger.warning("pynvml not available, using mock GPU metrics")
            metrics = self._get_mock_metrics()
        except Exception as e:
            logger.error("Failed to collect GPU metrics", error=str(e))
            metrics = self._get_mock_metrics()
        finally:
            if initialized:
                pynvml.nvmlShutdown()

        return metrics

    def _get_mock_metrics(self) -> dict[str, dict[str, Any]]:
        return {
            "gpu-0": {
                "utilization": 85.0,
                "memory_utilization": 87.5,
                "memory_used": 70 * 1024 * 1024 * 1024,
                "memory_total": 80 * 1024 * 1024 * 1024,
                "memory_free": 10 * 1024 * 1024 * 1024,
                "temperature": 75.0,
                "power_draw": 350.0,
                "power_limit": 400.0,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "gpu-1": {
                "utilization": 82.0,
                "memory_utilization": 85.0,
                "memory_used": 68 * 1024 * 1024 * 1024,
                "memory_total": 80 * 1024 * 1024 * 1024,
                "memory_free": 12 * 1024 * 1024 * 1024,
                "temperature": 72.0,
                "power_draw": 340.0,
                "power_limit": 400.0,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
