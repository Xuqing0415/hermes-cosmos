"""
Docker-based Device Simulator

Simulates multiple physical devices using Docker containers with
CPU limits and network delays for realistic testing.
"""

import docker
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulatedPhysicalDevice:
    """
    Simulates a physical device using Docker with realistic constraints.

    Features:
    - CPU limit simulation
    - Network latency simulation
    - Memory constraints
    - Real federated learning client
    """

    def __init__(self, device_id: str, cpu_limit: float = 0.5,
                 network_delay_ms: int = 50, image: str = "hermes-federated:latest"):
        """
        Initialize simulated device.

        Args:
            device_id: Unique device identifier
            cpu_limit: CPU limit (0.5 = half a CPU core)
            network_delay_ms: Simulated network delay in milliseconds
            image: Docker image to use
        """
        self.device_id = device_id
        self.cpu_limit = cpu_limit
        self.network_delay_ms = network_delay_ms
        self.image = image

        self.client = docker.from_env()
        self.container: Optional[docker.models.containers.Container] = None
        self.status = "stopped"
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0,
            'network_rx': 0,
            'network_tx': 0
        }

    def start(self, server_url: str, mount_path: str = None) -> bool:
        """
        Start the simulated device.

        Args:
            server_url: URL of the federated learning server
            mount_path: Path to mount for data exchange

        Returns:
            True if started successfully
        """
        try:
            mounts = []
            if mount_path:
                mounts.append(docker.types.Mount(
                    target="/app/data",
                    source=mount_path,
                    type="bind"
                ))

            self.container = self.client.containers.run(
                self.image,
                f"python -c 'from hermes_unified.physical_device import EdgeClient; "
                f"c = EdgeClient(\"{server_url}\", \"{self.device_id}\", [], 100); "
                f\"c.start(poll_interval=10, max_rounds=1000)'\"",
                detach=True,
                mem_limit=f"{int(512 * 1024 * 1024)}",  # 512MB
                cpu_period=100000,
                cpu_quota=int(100000 * self.cpu_limit),
                network_disabled=False,
                name=f"hermes-device-{self.device_id}",
                remove=True
            )

            self.status = "running"
            logger.info(f"Device {self.device_id} started")
            return True

        except docker.errors.DockerException as e:
            logger.error(f"Failed to start device {self.device_id}: {e}")
            return False

    def stop(self):
        """Stop the simulated device."""
        if self.container:
            try:
                self.container.stop(timeout=5)
            except:
                pass
            self.container = None

        self.status = "stopped"

    def get_logs(self, lines: int = 100) -> str:
        """Get device logs."""
        if self.container:
            try:
                return self.container.logs(tail=lines).decode('utf-8')
            except:
                return ""
        return ""

    def get_metrics(self) -> Dict[str, Any]:
        """Get device metrics."""
        if self.container:
            try:
                stats = self.container.stats(stream=False)
                cpu_stats = stats.get('cpu_stats', {})
                mem_stats = stats.get('memory_stats', {})

                self.metrics['cpu_usage'] = cpu_stats.get('cpu_usage', {}).get('percent', 0)
                self.metrics['memory_usage'] = mem_stats.get('usage', 0)
                self.metrics['network_rx'] = sum(
                    net.get('rx_bytes', 0)
                    for net in stats.get('networks', {}).values()
                )
                self.metrics['network_tx'] = sum(
                    net.get('tx_bytes', 0)
                    for net in stats.get('networks', {}).values()
                )
            except:
                pass

        return self.metrics.copy()

    def is_healthy(self) -> bool:
        """Check if device is healthy."""
        if not self.container:
            return False

        try:
            self.container.reload()
            return self.container.status == "running"
        except:
            return False


class DeviceSimulator:
    """
    Manages a cluster of simulated physical devices.

    Provides:
    - Device creation and management
    - Resource monitoring
    - Failure injection
    - Network delay simulation
    """

    def __init__(self, base_device_id: str = "device"):
        """
        Initialize device simulator.

        Args:
            base_device_id: Base name for devices
        """
        self.base_device_id = base_device_id
        self.devices: Dict[str, SimulatedPhysicalDevice] = {}
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.docker_client = docker.from_env()

    def create_devices(self, num_devices: int, cpu_limits: List[float] = None,
                     network_delays: List[int] = None) -> List[str]:
        """
        Create multiple simulated devices.

        Args:
            num_devices: Number of devices to create
            cpu_limits: List of CPU limits (one per device)
            network_delays: List of network delays (one per device)

        Returns:
            List of device IDs
        """
        if cpu_limits is None:
            cpu_limits = [0.5] * num_devices

        if network_delays is None:
            network_delays = [50] * num_devices

        device_ids = []

        for i in range(num_devices):
            device_id = f"{self.base_device_id}_{i}"

            device = SimulatedPhysicalDevice(
                device_id=device_id,
                cpu_limit=cpu_limits[i] if i < len(cpu_limits) else 0.5,
                network_delay_ms=network_delays[i] if i < len(network_delays) else 50
            )

            self.devices[device_id] = device
            device_ids.append(device_id)

        return device_ids

    def start_all(self, server_url: str, mount_path: str = None) -> int:
        """
        Start all simulated devices.

        Args:
            server_url: URL of federated learning server
            mount_path: Path to mount for data exchange

        Returns:
            Number of devices started successfully
        """
        started = 0

        for device in self.devices.values():
            if device.start(server_url, mount_path):
                started += 1
            time.sleep(0.5)  # Stagger startup

        return started

    def stop_all(self):
        """Stop all simulated devices."""
        for device in self.devices.values():
            device.stop()

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all devices."""
        return {
            device_id: device.get_metrics()
            for device_id, device in self.devices.items()
        }

    def get_device(self, device_id: str) -> SimulatedPhysicalDevice:
        """Get a specific device."""
        return self.devices.get(device_id)

    def get_healthy_devices(self) -> List[str]:
        """Get list of healthy device IDs."""
        return [
            device_id
            for device_id, device in self.devices.items()
            if device.is_healthy()
        ]

    def inject_failure(self, device_id: str, duration: int = 30):
        """
        Inject a failure into a device.

        Args:
            device_id: Device to inject failure into
            duration: Duration of failure in seconds
        """
        device = self.devices.get(device_id)
        if device and device.container:
            logger.info(f"Injecting failure into {device_id} for {duration}s")
            try:
                device.container.pause()
                time.sleep(duration)
                device.container.unpause()
            except:
                pass

    def inject_network_delay(self, device_id: str, delay_ms: int):
        """
        Inject network delay into a device (requires tc command).

        Args:
            device_id: Device to inject delay into
            delay_ms: Delay in milliseconds
        """
        device = self.devices.get(device_id)
        if device and device.container:
            try:
                device.container.exec_run(
                    f"tc qdisc add dev eth0 root netem delay {delay_ms}ms"
                )
                logger.info(f"Injected {delay_ms}ms delay into {device_id}")
            except:
                pass

    def start_monitoring(self, interval: int = 10):
        """Start background monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self, interval: int):
        """Background monitoring loop."""
        while self.running:
            metrics = self.get_all_metrics()

            healthy = self.get_healthy_devices()
            logger.info(f"Monitoring: {len(healthy)}/{len(self.devices)} devices healthy")

            time.sleep(interval)

    def cleanup(self):
        """Clean up all resources."""
        self.stop_monitoring()
        self.stop_all()

        for device in self.devices.values():
            try:
                if device.container:
                    device.container.remove(force=True)
            except:
                pass

        self.devices.clear()
        logger.info("Device simulator cleaned up")


def create_realistic_device_cluster(num_devices: int = 5) -> DeviceSimulator:
    """
    Create a realistic device cluster with varying capabilities.

    Args:
        num_devices: Number of devices to create

    Returns:
        Configured DeviceSimulator
    """
    np.random.seed(42)

    cpu_limits = np.random.uniform(0.2, 1.5, num_devices).tolist()
    network_delays = np.random.randint(20, 200, num_devices).tolist()

    simulator = DeviceSimulator()
    simulator.create_devices(num_devices, cpu_limits, network_delays)

    return simulator


if __name__ == "__main__":
    print("=== Testing Device Simulator ===")

    simulator = create_realistic_device_cluster(3)

    print(f"Created {len(simulator.devices)} simulated devices")

    for device_id, device in simulator.devices.items():
        print(f"  {device_id}: CPU={device.cpu_limit:.2f}, Delay={device.network_delay_ms}ms")

    print("\nNote: To start devices, run simulator.start_all('http://server:8000')")

    simulator.cleanup()
