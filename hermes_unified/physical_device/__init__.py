"""
Physical Device Integration Module

Enables Hermes to work with real physical devices for federated learning.
"""

from .device_coordinator import DeviceConnection, DeviceStatus, DeviceInfo, RealDeviceCoordinator
from .real_device_client import EdgeClient, TemperaturePredictionClient, ImageClassificationClient, create_temperature_data
from .federated_api_server import app, run_server, config

__all__ = [
    'DeviceConnection',
    'DeviceStatus',
    'DeviceInfo',
    'RealDeviceCoordinator',
    'EdgeClient',
    'TemperaturePredictionClient',
    'ImageClassificationClient',
    'create_temperature_data',
    'app',
    'run_server',
    'config'
]
