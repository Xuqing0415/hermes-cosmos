"""
Hermes Monitoring Dashboard

Real-time web dashboard for monitoring federated learning training progress.
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
import time
import numpy as np
from collections import defaultdict


app = Flask(__name__)
CORS(app)


class MonitoringDashboard:
    """
    In-memory metrics collector for the dashboard.
    """
    
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'participation_rate': [],
            'round_times': [],
            'communication_cost': []
        }
        
        self.device_stats = defaultdict(list)
        self.alerts = []
        self.lock = threading.Lock()
    
    def record_round(self, round_num: int, accuracy: float, loss: float,
                    participation: float, round_time: float, comm_cost: float):
        """Record metrics for a round."""
        with self.lock:
            self.metrics['accuracy'].append({'round': round_num, 'value': accuracy})
            self.metrics['loss'].append({'round': round_num, 'value': loss})
            self.metrics['participation_rate'].append({'round': round_num, 'value': participation})
            self.metrics['round_times'].append({'round': round_num, 'value': round_time})
            self.metrics['communication_cost'].append({'round': round_num, 'value': comm_cost})
    
    def record_device_stats(self, device_id: str, stats: dict):
        """Record device statistics."""
        with self.lock:
            self.device_stats[device_id].append({
                'timestamp': time.time(),
                'status': stats.get('status', 'unknown'),
                'battery': stats.get('battery_level', 0),
                'compute_speed': stats.get('compute_speed', 0)
            })
    
    def add_alert(self, message: str, severity: str = 'info'):
        """Add an alert."""
        with self.lock:
            self.alerts.append({
                'timestamp': time.time(),
                'message': message,
                'severity': severity
            })
    
    def get_metrics(self):
        """Get all metrics."""
        with self.lock:
            return dict(self.metrics)
    
    def get_device_stats(self):
        """Get device statistics."""
        with self.lock:
            return dict(self.device_stats)
    
    def get_alerts(self):
        """Get alerts."""
        with self.lock:
            return list(self.alerts)


dashboard = MonitoringDashboard()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hermes Federated Learning Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .device-list {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .device-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .status-online { background: #4caf50; color: white; }
        .status-offline { background: #f44336; color: white; }
        .status-training { background: #2196f3; color: white; }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-error { background: #f8d7da; border-left: 4px solid #dc3545; }
        .alert-info { background: #d1ecf1; border-left: 4px solid #17a2b8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Hermes Federated Learning Dashboard</h1>
        <p>Real-time monitoring of federated training</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Current Round</h3>
            <div class="metric-value" id="current-round">0</div>
        </div>
        <div class="metric-card">
            <h3>Accuracy</h3>
            <div class="metric-value" id="accuracy">0.00%</div>
        </div>
        <div class="metric-card">
            <h3>Participation Rate</h3>
            <div class="metric-value" id="participation">0%</div>
        </div>
        <div class="metric-card">
            <h3>Total Comm Cost</h3>
            <div class="metric-value" id="comm-cost">0 MB</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>Training Progress</h2>
        <div id="accuracy-chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>System Metrics</h2>
        <div id="metrics-chart"></div>
    </div>
    
    <div class="device-list">
        <h2>Connected Devices</h2>
        <div id="device-list">No devices connected</div>
    </div>
    
    <div class="alert-section">
        <h2>Alerts</h2>
        <div id="alerts"></div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metrics cards
                    const rounds = data.accuracy.length;
                    document.getElementById('current-round').textContent = rounds;
                    
                    if (rounds > 0) {
                        const lastAcc = data.accuracy[rounds - 1].value;
                        document.getElementById('accuracy').textContent = (lastAcc * 100).toFixed(2) + '%';
                        
                        const lastPart = data.participation_rate[rounds - 1].value;
                        document.getElementById('participation').textContent = (lastPart * 100).toFixed(0) + '%';
                        
                        const lastComm = data.communication_cost[rounds - 1].value;
                        document.getElementById('comm-cost').textContent = lastComm.toFixed(2) + ' MB';
                    }
                    
                    // Update accuracy chart
                    Plotly.newPlot('accuracy-chart', [{
                        x: data.accuracy.map(p => p.round),
                        y: data.accuracy.map(p => p.value * 100),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Accuracy (%)'
                    }], {
                        margin: {t: 0},
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent'
                    });
                });
            
            fetch('/api/devices')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('device-list');
                    if (data.length === 0) {
                        container.innerHTML = '<p>No devices connected</p>';
                    } else {
                        let html = '';
                        data.forEach(device => {
                            const statusClass = 'status-' + device.status.toLowerCase();
                            html += '<div class="device-item">';
                            html += '<span>' + device.device_id + ' (' + device.device_type + ')</span>';
                            html += '<span class="status-badge ' + statusClass + '">' + device.status + '</span>';
                            html += '</div>';
                        });
                        container.innerHTML = html;
                    }
                });
            
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('alerts');
                    if (data.length === 0) {
                        container.innerHTML = '<p>No alerts</p>';
                    } else {
                        let html = '';
                        data.slice(-5).forEach(alert => {
                            html += '<div class="alert alert-' + alert.severity + '">';
                            html += '<strong>' + alert.severity.toUpperCase() + '</strong>: ';
                            html += alert.message;
                            html += '</div>';
                        });
                        container.innerHTML = html;
                    }
                });
        }
        
        // Update dashboard every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Dashboard home page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics."""
    return jsonify(dashboard.get_metrics())


@app.route('/api/devices')
def get_devices():
    """Get device statistics."""
    return jsonify(dashboard.get_device_stats())


@app.route('/api/alerts')
def get_alerts():
    """Get alerts."""
    return jsonify(dashboard.get_alerts())


@app.route('/api/record', methods=['POST'])
def record_metrics():
    """Record new metrics."""
    data = request.json
    
    dashboard.record_round(
        round_num=data.get('round', 0),
        accuracy=data.get('accuracy', 0.0),
        loss=data.get('loss', 0.0),
        participation=data.get('participation', 0.0),
        round_time=data.get('round_time', 0.0),
        comm_cost=data.get('comm_cost', 0.0)
    )
    
    return jsonify({'status': 'ok'})


def run_dashboard(host='0.0.0.0', port=8501):
    """Run the monitoring dashboard."""
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    print("Starting Hermes Monitoring Dashboard on http://localhost:8501")
    run_dashboard()
