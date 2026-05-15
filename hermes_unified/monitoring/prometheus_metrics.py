"""
Prometheus Metrics Integration

Exposes federated learning metrics for Prometheus monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import time
import threading


class FederatedMetrics:
    """
    Prometheus metrics collector for federated learning.
    
    Exposes:
    - round_duration_seconds: Time taken for each federated round
    - global_accuracy: Current global model accuracy
    - active_clients_count: Number of active clients per round
    - communication_bytes_total: Total bytes communicated
    - attack_events_total: Count of detected attacks
    - defense_effectiveness: Effectiveness score of defense mechanisms
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single metrics instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        if self._initialized:
            return
        
        self._initialized = True
        
        # Round metrics
        self.round_duration = Histogram(
            'hermes_round_duration_seconds',
            'Time taken for each federated round',
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        
        self.round_counter = Counter(
            'hermes_rounds_total',
            'Total number of federated rounds completed'
        )
        
        # Model metrics
        self.global_accuracy = Gauge(
            'hermes_global_accuracy',
            'Current global model accuracy'
        )
        
        self.global_loss = Gauge(
            'hermes_global_loss',
            'Current global model loss'
        )
        
        self.model_norm = Gauge(
            'hermes_model_norm',
            'L2 norm of the global model'
        )
        
        # Client metrics
        self.active_clients = Gauge(
            'hermes_active_clients_count',
            'Number of active clients in current round'
        )
        
        self.total_clients = Gauge(
            'hermes_total_clients_count',
            'Total number of registered clients'
        )
        
        self.participation_rate = Gauge(
            'hermes_participation_rate',
            'Ratio of active to total clients'
        )
        
        # Communication metrics
        self.communication_bytes = Counter(
            'hermes_communication_bytes_total',
            'Total bytes communicated between server and clients'
        )
        
        self.update_size_bytes = Histogram(
            'hermes_update_size_bytes',
            'Size of model updates in bytes',
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600]
        )
        
        # Attack and defense metrics
        self.attack_events = Counter(
            'hermes_attack_events_total',
            'Total number of detected attack events',
            ['attack_type']
        )
        
        self.defense_effectiveness = Gauge(
            'hermes_defense_effectiveness',
            'Effectiveness score of the defense mechanism (0-1)'
        )
        
        self.aggregated_updates = Counter(
            'hermes_aggregated_updates_total',
            'Total number of successfully aggregated updates'
        )
        
        # Error metrics
        self.client_failures = Counter(
            'hermes_client_failures_total',
            'Total number of client failures',
            ['failure_type']
        )
        
        self.aggregation_errors = Counter(
            'hermes_aggregation_errors_total',
            'Total number of aggregation errors'
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'hermes_cpu_usage_percent',
            'CPU usage of the server process'
        )
        
        self.memory_usage = Gauge(
            'hermes_memory_usage_bytes',
            'Memory usage of the server process in bytes'
        )
    
    def record_round(self, duration: float, accuracy: float, loss: float,
                   active_clients: int, total_clients: int, comm_bytes: float):
        """Record metrics for a completed round."""
        self.round_duration.observe(duration)
        self.round_counter.inc()
        self.global_accuracy.set(accuracy)
        self.global_loss.set(loss)
        self.active_clients.set(active_clients)
        self.total_clients.set(total_clients)
        self.participation_rate.set(active_clients / total_clients if total_clients > 0 else 0)
        self.communication_bytes.inc(comm_bytes)
    
    def record_attack(self, attack_type: str):
        """Record a detected attack."""
        self.attack_events.labels(attack_type=attack_type).inc()
    
    def record_defense_effectiveness(self, effectiveness: float):
        """Record defense effectiveness score."""
        self.defense_effectiveness.set(effectiveness)
    
    def record_client_failure(self, failure_type: str):
        """Record a client failure."""
        self.client_failures.labels(failure_type=failure_type).inc()
    
    def record_aggregation_error(self):
        """Record an aggregation error."""
        self.aggregation_errors.inc()
    
    def record_update_size(self, size_bytes: int):
        """Record the size of a model update."""
        self.update_size_bytes.observe(size_bytes)
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int):
        """Update system resource usage metrics."""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_bytes)


class MetricsCollector:
    """
    Background thread for collecting and exporting system metrics.
    """
    
    def __init__(self, interval: int = 10):
        """
        Initialize metrics collector.
        
        Args:
            interval: Collection interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics = FederatedMetrics()
    
    def start(self, port: int = 9090):
        """
        Start the metrics collector and HTTP server.
        
        Args:
            port: Port to expose metrics on
        """
        self.running = True
        start_http_server(port)
        
        self.thread = threading.Thread(target=self._collect_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"Metrics server started on port {port}")
    
    def stop(self):
        """Stop the metrics collector."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _collect_loop(self):
        """Background loop for collecting system metrics."""
        import psutil
        
        while self.running:
            try:
                process = psutil.Process()
                cpu = process.cpu_percent(interval=1)
                memory = process.memory_info().rss
                self.metrics.update_system_metrics(cpu, memory)
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
            
            time.sleep(self.interval)


# Global metrics instance
metrics = FederatedMetrics()


if __name__ == "__main__":
    print("Starting Prometheus metrics server on port 9090...")
    
    collector = MetricsCollector(interval=10)
    collector.start(port=9090)
    
    print("Metrics available at http://localhost:9090/metrics")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping metrics collector...")
        collector.stop()
