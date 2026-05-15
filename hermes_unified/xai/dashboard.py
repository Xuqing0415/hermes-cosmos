"""
FedXAI Dashboard: Visualization for Federated Explainability

Web dashboard for visualizing feature importance, client comparisons,
and anomaly detection in federated learning models.
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class FedXAIDashboard:
    """
    In-memory data store for the XAI dashboard.
    """
    
    def __init__(self):
        self.global_importance = []
        self.client_importance = {}
        self.round_history = []
        self.anomaly_alerts = []
        self.feature_names = []
        self.lock = threading.Lock()
    
    def update(self, data: Dict[str, Any]):
        """Update dashboard with new data."""
        with self.lock:
            if 'global_importance' in data:
                self.global_importance = data['global_importance']
            
            if 'client_importance' in data:
                self.client_importance = data['client_importance']
            
            if 'feature_names' in data:
                self.feature_names = data['feature_names']
            
            if 'history' in data:
                self.round_history.extend(data['history'])
            
            if 'alerts' in data:
                self.anomaly_alerts.extend(data['alerts'])
    
    def get_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        with self.lock:
            return {
                'global_importance': self.global_importance,
                'client_importance': self.client_importance,
                'feature_names': self.feature_names,
                'round_history': self.round_history[-50:],  # Last 50 rounds
                'anomaly_alerts': self.anomaly_alerts[-20:],  # Last 20 alerts
                'n_clients': len(self.client_importance),
                'n_rounds': len(self.round_history),
                'n_alerts': len(self.anomaly_alerts)
            }


dashboard = FedXAIDashboard()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FedXAI - Federated Explainable AI Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        
        .stat-value.warning { color: #f39c12; }
        .stat-value.danger { color: #e74c3c; }
        .stat-value.success { color: #27ae60; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-container h2 {
            color: #333;
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .alert-list {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        
        .alert-item {
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid;
        }
        
        .alert-item.high {
            background: #ffe6e6;
            border-color: #e74c3c;
        }
        
        .alert-item.medium {
            background: #fff3e6;
            border-color: #f39c12;
        }
        
        .alert-item.low {
            background: #e6f7ff;
            border-color: #3498db;
        }
        
        .alert-item strong {
            color: #333;
        }
        
        .alert-item span {
            color: #666;
            font-size: 12px;
        }
        
        .heatmap-container {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .progress-bar {
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>FedXAI Dashboard</h1>
        <p>Federated Explainable AI - Feature Importance & Anomaly Detection</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <h3>Total Clients</h3>
            <div class="stat-value" id="n-clients">0</div>
        </div>
        <div class="stat-card">
            <h3>Training Rounds</h3>
            <div class="stat-value" id="n-rounds">0</div>
        </div>
        <div class="stat-card">
            <h3>Total Explanations</h3>
            <div class="stat-value" id="n-explanations">0</div>
        </div>
        <div class="stat-card">
            <h3>Anomaly Alerts</h3>
            <div class="stat-value" id="n-alerts">0</div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-container">
            <h2>Global Feature Importance</h2>
            <div id="importance-chart"></div>
        </div>
        
        <div class="chart-container">
            <h2>Importance Over Time</h2>
            <div id="time-chart"></div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-container">
            <h2>Client Feature Importance Heatmap</h2>
            <div id="heatmap-chart" class="heatmap-container"></div>
        </div>
        
        <div class="chart-container">
            <h2>Cross-Client Variance</h2>
            <div id="variance-chart"></div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-container">
            <h2>Anomaly Detection Alerts</h2>
            <div id="alerts-list" class="alert-list">
                <p class="loading">No alerts</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Top Features by Client</h2>
            <div id="top-features-chart"></div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/xai/data')
                .then(response => response.json())
                .then(data => {
                    updateStats(data);
                    updateCharts(data);
                    updateAlerts(data);
                })
                .catch(err => console.error('Error fetching data:', err));
        }
        
        function updateStats(data) {
            document.getElementById('n-clients').textContent = data.n_clients || 0;
            document.getElementById('n-rounds').textContent = data.n_rounds || 0;
            document.getElementById('n-explanations').textContent = data.n_rounds * data.n_clients || 0;
            
            const alertEl = document.getElementById('n-alerts');
            alertEl.textContent = data.n_alerts || 0;
            alertEl.className = 'stat-value' + (data.n_alerts > 5 ? ' danger' : data.n_alerts > 0 ? ' warning' : ' success');
        }
        
        function updateCharts(data) {
            // Feature Importance Bar Chart
            if (data.global_importance && data.global_importance.length > 0) {
                const featureNames = data.feature_names && data.feature_names.length > 0 
                    ? data.feature_names.slice(0, data.global_importance.length)
                    : data.global_importance.map((_, i) => 'Feature ' + i);
                
                const trace = [{
                    type: 'bar',
                    x: data.global_importance,
                    y: featureNames,
                    orientation: 'h',
                    marker: {
                        color: data.global_importance,
                        colorscale: 'Viridis'
                    }
                }];
                
                Plotly.newPlot('importance-chart', trace, {
                    margin: {l: 100, r: 20, t: 20, b: 40},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                });
            }
            
            // Time Series Chart
            if (data.round_history && data.round_history.length > 0) {
                const rounds = data.round_history.map(r => r.round);
                const means = data.round_history.map(r => {
                    if (Array.isArray(r.mean)) return r.mean.reduce((a, b) => a + b, 0) / r.mean.length;
                    return 0;
                });
                
                const trace = [{
                    x: rounds,
                    y: means,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Mean Importance',
                    line: {color: '#667eea', width: 2}
                }];
                
                Plotly.newPlot('time-chart', trace, {
                    margin: {l: 50, r: 20, t: 20, b: 40},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                });
            }
            
            // Heatmap
            if (data.client_importance) {
                const clientIds = Object.keys(data.client_importance);
                const featureNames = data.feature_names && data.feature_names.length > 0
                    ? data.feature_names
                    : ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'];
                
                const z = clientIds.map(cid => {
                    const imp = data.client_importance[cid];
                    return imp.slice(0, Math.min(featureNames.length, imp.length));
                });
                
                const trace = [{
                    z: z,
                    x: featureNames.slice(0, z[0]?.length || 10),
                    y: clientIds,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }];
                
                Plotly.newPlot('heatmap-chart', trace, {
                    margin: {l: 100, r: 20, t: 20, b: 60},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                });
            }
            
            // Variance Chart
            if (data.round_history && data.round_history.length > 0) {
                const rounds = data.round_history.map(r => r.round);
                const stds = data.round_history.map(r => {
                    if (Array.isArray(r.std)) return r.std.reduce((a, b) => a + b, 0) / r.std.length;
                    return 0;
                });
                
                const trace = [{
                    x: rounds,
                    y: stds,
                    type: 'bar',
                    name: 'Variance',
                    marker: {color: '#e74c3c'}
                }];
                
                Plotly.newPlot('variance-chart', trace, {
                    margin: {l: 50, r: 20, t: 20, b: 40},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                });
            }
            
            // Top Features by Client
            if (data.client_importance) {
                const topFeatures = [];
                
                for (const [clientId, importance] of Object.entries(data.client_importance)) {
                    const maxIdx = importance.indexOf(Math.max(...importance));
                    const featureName = data.feature_names && data.feature_names[maxIdx] 
                        ? data.feature_names[maxIdx] 
                        : 'Feature ' + maxIdx;
                    topFeatures.push({client: clientId, feature: featureName, importance: importance[maxIdx]});
                }
                
                const trace = [{
                    x: topFeatures.map(f => f.client),
                    y: topFeatures.map(f => f.importance),
                    text: topFeatures.map(f => f.feature),
                    type: 'bar',
                    marker: {color: '#27ae60'}
                }];
                
                Plotly.newPlot('top-features-chart', trace, {
                    margin: {l: 50, r: 20, t: 20, b: 60},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                });
            }
        }
        
        function updateAlerts(data) {
            const container = document.getElementById('alerts-list');
            
            if (!data.anomaly_alerts || data.anomaly_alerts.length === 0) {
                container.innerHTML = '<p class="loading">No alerts - all clients behaving normally</p>';
                return;
            }
            
            let html = '';
            for (const alert of data.anomaly_alerts.slice(-10).reverse()) {
                const severity = alert.severity || 'medium';
                html += `
                    <div class="alert-item ${severity}">
                        <strong>${alert.client_id}</strong>
                        <br>
                        <span>Deviation: ${(alert.deviation || 0).toFixed(4)}</span>
                        <span> - Severity: ${severity.toUpperCase()}</span>
                        <br>
                        <span>${new Date(alert.timestamp * 1000).toLocaleString()}</span>
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        // Auto-refresh every 5 seconds
        setInterval(updateDashboard, 5000);
        updateDashboard();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Dashboard home page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/xai/data')
def get_xai_data():
    """Get XAI dashboard data."""
    return jsonify(dashboard.get_data())


@app.route('/api/xai/update', methods=['POST'])
def update_xai_data():
    """Update dashboard with new XAI data."""
    data = request.json
    dashboard.update(data)
    return jsonify({'status': 'ok'})


@app.route('/api/xai/anomalies')
def get_anomalies():
    """Get anomaly alerts."""
    return jsonify(dashboard.get_data()['anomaly_alerts'])


def run_xai_dashboard(host: str = '0.0.0.0', port: int = 8502):
    """
    Run the FedXAI dashboard server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    print(f"Starting FedXAI Dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_xai_dashboard()
