"""
Self-Evolving FL Runner

Main entry point for self-evolution.
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from self_evolution import SelfEvolvingFederatedLearning


def run_self_evolution(num_generations: int = 10,
                       max_time_hours: float = 1.0,
                       start_server: bool = True,
                       server_port: int = 8504):
    """
    Run self-evolution.
    
    Args:
        num_generations: Number of generations
        max_time_hours: Maximum time in hours
        start_server: Whether to start the dashboard server
        server_port: Dashboard port
    """
    print("=" * 60)
    print("Self-Evolving Federated Learning")
    print("=" * 60)
    
    # Create self-evolving system
    sef = SelfEvolvingFederatedLearning()
    
    # Start server if requested
    if start_server:
        import threading
        server_thread = threading.Thread(
            target=start_dashboard_server,
            args=(sef, server_port),
            daemon=True
        )
        server_thread.start()
        
        print(f"\n📊 Dashboard available at: http://localhost:{server_port}")
        time.sleep(2)
    
    # Run evolution
    sef.evolve(num_generations=num_generations,
                max_time_hours=max_time_hours)
    
    # Generate final report
    print("\nGenerating final report...")
    report = sef.generate_evolution_report()
    
    print("\n" + "=" * 60)
    print("Evolution Complete! Final Report")
    print("=" * 60)
    
    print(f"\nTotal Experiments: {report['summary']['total_experiments']}")
    print(f"Generations: {report['summary']['generations']}")
    print(f"Best Accuracy: {report['summary']['best_accuracy']:.4f}")
    print(f"Milestones: {report['summary']['milestones']}")
    
    print("\nAlgorithm Performance Summary:")
    for algo, stats in report['algorithm_statistics'].items():
        print(f"  {algo}: {stats['mean_accuracy']:.4f} (n={stats['count']})")
    
    if report['best_configuration']:
        print("\nBest Configuration Found:")
        print(f"  Algorithm: {report['best_configuration']['algorithm']}")
        print(f"  Clients: {report['best_configuration']['num_clients']}")
        print(f"  Non-IID Level: {report['best_configuration']['non_iid_level']:.2f}")
    
    if start_server:
        print("\nPress Ctrl+C to stop the dashboard server...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")


def start_dashboard_server(sef: SelfEvolvingFederatedLearning, port: int):
    """Start the evolution dashboard server."""
    from flask import Flask, jsonify, render_template_string
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/evolution/tree')
    def get_evolution_tree():
        tree = sef.get_evolution_tree()
        return jsonify(tree)
    
    @app.route('/api/evolution/report')
    def get_evolution_report():
        report = sef.generate_evolution_report()
        return jsonify(report)
    
    @app.route('/api/evolution/progress')
    def get_progress():
        with open(os.path.join(sef.output_dir, 'evolution_progress.json'), 'r') as f:
            return jsonify(json.load(f))
    
    @app.route('/')
    def index():
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Self-Evolving FL Dashboard</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; min-height: 100vh; }
                .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
                h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
                .stat-card h3 { font-size: 14px; opacity: 0.9; margin-bottom: 10px; }
                .stat-value { font-size: 28px; font-weight: bold; }
                .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .chart-container { background: #f8f9fa; padding: 20px; border-radius: 8px; }
                .chart-container h2 { color: #333; font-size: 18px; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
                .milestone { background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; border-radius: 4px; }
                .loading { text-align: center; padding: 40px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Self-Evolving FL Dashboard</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Experiments</h3>
                        <div class="stat-value" id="total-experiments">0</div>
                    </div>
                    <div class="stat-card">
                        <h3>Generations</h3>
                        <div class="stat-value" id="generations">0</div>
                    </div>
                    <div class="stat-card">
                        <h3>Best Accuracy</h3>
                        <div class="stat-value" id="best-accuracy">0.0000</div>
                    </div>
                    <div class="stat-card">
                        <h3>Milestones</h3>
                        <div class="stat-value" id="milestones">0</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-container">
                        <h2>Evolution Tree</h2>
                        <div id="evolution-tree"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h2>Accuracy Trend</h2>
                        <div id="accuracy-trend"></div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-container">
                        <h2>Algorithm Performance</h2>
                        <div id="algorithm-perf"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h2>Recent Milestones</h2>
                        <div id="milestones-list"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // Auto-refresh
                function refreshData();
                setInterval(refreshData, 5000);
                
                async function refreshData() {
                    await Promise.all([
                    loadTree(), loadReport()]);
                }
                
                async function loadTree() {
                    const response = await fetch('/api/evolution/tree');
                    const tree = await response.json();
                    renderTree(tree);
                }
                
                async function loadReport() {
                    const response = await fetch('/api/evolution/report');
                    const report = await response.json();
                    renderReport(report);
                }
                
                function renderTree(tree) {
                    if (!tree || tree.length === 0) return;
                    
                    // Simple tree visualization
                    const exps = tree.slice(-30);
                    
                    const colors = { fedavg: '#2c3e50, ditto: '#e74c3c, fedrep: '#27ae60, krum: '#9b59b6, trimmed_mean: '#3498db};
                    const algs = exps.map((e, i) => ({x: e.algorithm});
                    const ids = exps.map(e => e.experiment_id);
                    const accs = exps.map(e => e.actual_accuracy);
                    
                    const trace = [{
                        type: 'scatter',
                        mode: 'markers',
                        x: ids,
                        y: accs,
                        marker: { size: 12,
                            color: algs.map(a => colors[a]),
                            opacity: 0.8
                        },
                        text: exps.map(e => `${e.algorithm}: ${e.actual_accuracy.toFixed(4)}`)
                    }];
                    
                    Plotly.newPlot('evolution-tree', trace, {
                        title: '',
                        xaxis: {title: 'Experiment ID'},
                        yaxis: {title: 'Accuracy', range: [0.5, 1.0},
                        showlegend: false
                    });
                }
                
                function renderReport(report) {
                    // Update stats
                    document.getElementById('total-experiments').textContent = report.summary.total_experiments;
                    document.getElementById('generations').textContent = report.summary.generations;
                    document.getElementById('best-accuracy').textContent = report.summary.best_accuracy.toFixed(4);
                    document.getElementById('milestones').textContent = report.summary.milestones;
                    
                    // Accuracy trend
                    if (report.accuracy_trend) {
                        const trace = [{
                            type: 'line',
                            x: report.accuracy_trend.map((_, i) => i),
                            y: report.accuracy_trend
                        }];
                        
                        Plotly.newPlot('accuracy-trend', trace, {
                            title: '',
                            xaxis: {title: 'Recent Experiments'},
                            yaxis: {title: 'Accuracy', range: [0.5, 1.0}
                        });
                    }
                    
                    // Algorithm performance
                    if (report.algorithm_statistics) {
                        const algos = Object.keys(report.algorithm_statistics);
                        const means = algos.map(a => report.algorithm_statistics[a].mean_accuracy);
                        const counts = algos.map(a => report.algorithm_statistics[a].count);
                        
                        const trace = [{
                            type: 'bar',
                            x: algos,
                            y: means,
                            text: counts.map(c => `n=${c}),
                            marker: {color: ['#2c3e50', '#e74c3c', '#27ae60', '#9b59b6', '#3498db']
                        }];
                        
                        Plotly.newPlot('algorithm-perf', trace, {
                            title: '',
                            yaxis: {range: [0.5, 1.0}
                        });
                    }
                    
                    // Milestones list
                    const milestoneDiv = document.getElementById('milestones-list');
                    if (report.milestones && report.milestones.length > 0) {
                        const html = report.milestones.slice(-5).map(id => 
                            `<div class="milestone">Experiment ${id} was a milestone!</div>`
                        ).join('');
                        milestoneDiv.innerHTML = html;
                    } else {
                        milestoneDiv.innerHTML = '<div class="loading">No milestones yet...</div>';
                    }
                }
            </script>
        </body>
        </html>
        """)
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-Evolving Federated Learning')
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--hours', type=float, default=1.0)
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--port', type=int, default=8504)
    
    args = parser.parse_args()
    
    run_self_evolution(
        num_generations=args.generations,
        max_time_hours=args.hours,
        start_server=not args.no_server,
        server_port=args.port
    )
