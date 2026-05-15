"""
Real Device Integration Module

Enables Hermes to work with real physical devices (Raspberry Pi, ESP32, etc.)
for practical federated learning tasks.
"""

import socket
import json
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class DeviceStatus(Enum):
    """Device status enumeration."""
    OFFLINE = "offline"
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    ERROR = "error"


@dataclass
class DeviceInfo:
    """Information about a connected device."""
    device_id: str
    device_type: str
    ip_address: str
    port: int
    compute_speed: float
    memory_mb: int
    battery_level: float
    status: DeviceStatus
    last_seen: float
    training_stats: Dict[str, Any]


class DeviceConnection:
    """
    Manages a connection to a physical device.
    
    Handles model distribution, update collection, and health monitoring.
    """
    
    def __init__(self, device_id: str, host: str, port: int, device_type: str = "generic"):
        """
        Initialize device connection.
        
        Args:
            device_id: Unique device identifier
            host: Device IP address or hostname
            port: Device port number
            device_type: Type of device (rpi, esp32, android, etc.)
        """
        self.device_id = device_id
        self.host = host
        self.port = port
        self.device_type = device_type
        
        self.socket: Optional[socket.socket] = None
        self.status = DeviceStatus.OFFLINE
        self.last_heartbeat = time.time()
        
        # Device capabilities (will be updated after handshake)
        self.compute_speed = 1.0
        self.memory_mb = 512
        self.battery_level = 100.0
        self.model_size_limit_mb = 100
        
        # Statistics
        self.training_stats = {
            'rounds_completed': 0,
            'total_training_time': 0.0,
            'updates_sent': 0,
            'updates_received': 0,
            'last_round_accuracy': 0.0
        }
    
    def connect(self, timeout: float = 5.0) -> bool:
        """
        Establish connection to the device.
        
        Args:
            timeout: Connection timeout in seconds
        
        Returns:
            True if connected successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.status = DeviceStatus.IDLE
            self.last_heartbeat = time.time()
            
            # Perform handshake
            self._send_message({'type': 'handshake', 'device_id': self.device_id})
            response = self._receive_message()
            
            if response and response.get('type') == 'handshake_ack':
                self._update_device_info(response.get('device_info', {}))
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to connect to {self.device_id}: {e}")
            self.status = DeviceStatus.OFFLINE
            return False
    
    def disconnect(self):
        """Close the connection to the device."""
        if self.socket:
            try:
                self._send_message({'type': 'disconnect'})
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
                self.status = DeviceStatus.OFFLINE
    
    def _update_device_info(self, info: Dict[str, Any]):
        """Update device information after handshake."""
        self.compute_speed = info.get('compute_speed', 1.0)
        self.memory_mb = info.get('memory_mb', 512)
        self.battery_level = info.get('battery_level', 100.0)
        self.model_size_limit_mb = info.get('model_size_limit_mb', 100)
    
    def send_model(self, model_data: bytes, compressed: bool = True) -> bool:
        """
        Send model to the device.
        
        Args:
            model_data: Model parameters as bytes
            compressed: Whether the model is compressed
        
        Returns:
            True if sent successfully
        """
        if self.status == DeviceStatus.OFFLINE:
            return False
        
        try:
            self.status = DeviceStatus.DOWNLOADING
            self._send_message({
                'type': 'model',
                'data_size': len(model_data),
                'compressed': compressed,
                'timestamp': time.time()
            })
            
            # Send model data in chunks
            chunk_size = 8192
            for i in range(0, len(model_data), chunk_size):
                self.socket.sendall(model_data[i:i+chunk_size])
            
            # Wait for acknowledgment
            response = self._receive_message()
            self.status = DeviceStatus.IDLE
            
            return response and response.get('type') == 'model_ack'
            
        except Exception as e:
            print(f"Failed to send model to {self.device_id}: {e}")
            self.status = DeviceStatus.ERROR
            return False
    
    def receive_update(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        Receive model update from the device.
        
        Args:
            timeout: Receive timeout in seconds
        
        Returns:
            Model update as bytes, or None if failed
        """
        if self.status == DeviceStatus.OFFLINE:
            return None
        
        try:
            self.status = DeviceStatus.UPLOADING
            self.socket.settimeout(timeout)
            
            # Wait for update message
            message = self._receive_message()
            
            if message and message.get('type') == 'update':
                data_size = message.get('data_size', 0)
                
                # Receive update data
                data = b''
                while len(data) < data_size:
                    chunk = self.socket.recv(min(8192, data_size - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                # Send acknowledgment
                self._send_message({'type': 'update_ack'})
                
                self.training_stats['updates_received'] += 1
                self.status = DeviceStatus.IDLE
                return data
            
            self.status = DeviceStatus.IDLE
            return None
            
        except Exception as e:
            print(f"Failed to receive update from {self.device_id}: {e}")
            self.status = DeviceStatus.ERROR
            return None
    
    def send_heartbeat(self) -> bool:
        """
        Send heartbeat to check device health.
        
        Returns:
            True if device is responsive
        """
        if not self.socket or self.status == DeviceStatus.OFFLINE:
            return False
        
        try:
            self._send_message({'type': 'heartbeat', 'timestamp': time.time()})
            response = self._receive_message(timeout=2.0)
            
            if response and response.get('type') == 'heartbeat_ack':
                self.last_heartbeat = time.time()
                self.battery_level = response.get('battery_level', self.battery_level)
                return True
            
            return False
            
        except:
            self.status = DeviceStatus.OFFLINE
            return False
    
    def _send_message(self, message: Dict[str, Any]):
        """Send a JSON message to the device."""
        if not self.socket:
            raise RuntimeError("Not connected")
        
        data = json.dumps(message).encode('utf-8')
        length = len(data)
        self.socket.sendall(length.to_bytes(4, 'big') + data)
    
    def _receive_message(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive a JSON message from the device."""
        if not self.socket:
            return None
        
        try:
            self.socket.settimeout(timeout)
            
            # Read message length
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = self.socket.recv(4 - len(length_bytes))
                if not chunk:
                    return None
                length_bytes += chunk
            
            length = int.from_bytes(length_bytes, 'big')
            
            # Read message data
            data = b''
            while len(data) < length:
                chunk = self.socket.recv(min(8192, length - len(data)))
                if not chunk:
                    return None
                data += chunk
            
            return json.loads(data.decode('utf-8'))
            
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    def get_info(self) -> DeviceInfo:
        """Get device information."""
        return DeviceInfo(
            device_id=self.device_id,
            device_type=self.device_type,
            ip_address=self.host,
            port=self.port,
            compute_speed=self.compute_speed,
            memory_mb=self.memory_mb,
            battery_level=self.battery_level,
            status=self.status,
            last_seen=self.last_heartbeat,
            training_stats=self.training_stats.copy()
        )


class RealDeviceCoordinator:
    """
    Coordinator for managing real physical devices.
    
    Provides high-level APIs for distributing models and collecting updates.
    """
    
    def __init__(self, server_host: str = "0.0.0.0", server_port: int = 8000):
        """
        Initialize the coordinator.
        
        Args:
            server_host: Host address for the coordinator server
            server_port: Port for the coordinator server
        """
        self.server_host = server_host
        self.server_port = server_port
        
        self.devices: Dict[str, DeviceConnection] = {}
        self.device_lock = threading.Lock()
        
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_rounds': 0,
            'total_updates': 0,
            'failed_updates': 0,
            'average_round_time': 0.0
        }
    
    def start_server(self):
        """Start the device coordinator server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.server_host, self.server_port))
        self.server_socket.listen(5)
        
        self.running = True
        
        # Start heartbeat checker
        heartbeat_thread = threading.Thread(target=self._heartbeat_checker)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        print(f"Device coordinator listening on {self.server_host}:{self.server_port}")
    
    def stop_server(self):
        """Stop the coordinator server."""
        self.running = False
        
        with self.device_lock:
            for device in self.devices.values():
                device.disconnect()
            self.devices.clear()
        
        if self.server_socket:
            self.server_socket.close()
    
    def accept_connections(self):
        """Accept incoming device connections (call in main loop)."""
        if not self.server_socket:
            return
        
        try:
            self.server_socket.settimeout(1.0)
            client_socket, address = self.server_socket.accept()
            
            # Handle connection in separate thread
            thread = threading.Thread(target=self._handle_connection, args=(client_socket, address))
            thread.daemon = True
            thread.start()
            
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error accepting connection: {e}")
    
    def _handle_connection(self, client_socket: socket.socket, address: tuple):
        """Handle a new device connection."""
        try:
            # Receive handshake
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = client_socket.recv(4 - len(length_bytes))
                if not chunk:
                    return
                length_bytes += chunk
            
            length = int.from_bytes(length_bytes, 'big')
            data = b''
            while len(data) < length:
                chunk = client_socket.recv(min(8192, length - len(data)))
                if not chunk:
                    return
                data += chunk
            
            message = json.loads(data.decode('utf-8'))
            
            if message.get('type') == 'register':
                device_id = message.get('device_id', f"device_{address[0]}_{address[1]}")
                
                device = DeviceConnection(
                    device_id=device_id,
                    host=address[0],
                    port=address[1],
                    device_type=message.get('device_type', 'unknown')
                )
                device.socket = client_socket
                device.status = DeviceStatus.IDLE
                device.last_heartbeat = time.time()
                
                with self.device_lock:
                    self.devices[device_id] = device
                
                # Send registration acknowledgment
                device._send_message({
                    'type': 'register_ack',
                    'server_time': time.time()
                })
                
                print(f"Device registered: {device_id} from {address}")
            
        except Exception as e:
            print(f"Error handling connection: {e}")
            client_socket.close()
    
    def _heartbeat_checker(self):
        """Periodically check device health."""
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            
            with self.device_lock:
                offline_devices = []
                
                for device_id, device in self.devices.items():
                    if not device.send_heartbeat():
                        offline_devices.append(device_id)
                
                for device_id in offline_devices:
                    self.devices[device_id].status = DeviceStatus.OFFLINE
                    print(f"Device offline: {device_id}")
    
    def get_online_devices(self) -> list:
        """Get list of online devices."""
        with self.device_lock:
            return [d.get_info() for d in self.devices.values() if d.status != DeviceStatus.OFFLINE]
    
    def get_all_devices(self) -> list:
        """Get list of all registered devices."""
        with self.device_lock:
            return [d.get_info() for d in self.devices.values()]
    
    def distribute_model(self, model_data: bytes) -> int:
        """
        Distribute model to all online devices.
        
        Args:
            model_data: Model parameters as bytes
        
        Returns:
            Number of successful distributions
        """
        success_count = 0
        
        with self.device_lock:
            for device in self.devices.values():
                if device.status != DeviceStatus.OFFLINE:
                    if device.send_model(model_data):
                        success_count += 1
        
        return success_count
    
    def collect_updates(self, timeout: float = 30.0) -> list:
        """
        Collect model updates from devices.
        
        Args:
            timeout: Timeout for each device in seconds
        
        Returns:
            List of (device_id, update_data) tuples
        """
        updates = []
        
        with self.device_lock:
            for device_id, device in self.devices.items():
                if device.status == DeviceStatus.IDLE:
                    update = device.receive_update(timeout=timeout)
                    if update:
                        updates.append((device_id, update))
        
        return updates


if __name__ == "__main__":
    print("=== Testing Real Device Coordinator ===")
    
    coordinator = RealDeviceCoordinator(server_host="0.0.0.0", server_port=9000)
    coordinator.start_server()
    
    print("Coordinator started. Waiting for devices...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            coordinator.accept_connections()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping coordinator...")
        coordinator.stop_server()
