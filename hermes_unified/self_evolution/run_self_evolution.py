"""
Self-Evolving FL Runner

Main entry point for self-evolution.

[NOTE] 本模块运行的是模拟实验（simulation），不是真实联邦训练。
       SelfEvolvingFederatedLearning.run_single_experiment 用合成任务与带噪声的
       预测值代替真实 FL 训练，因此这里输出的精度/轮次只是演示数据，不能作为
       真实联邦学习能力的证据。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_evolution import SelfEvolvingFederatedLearning  # noqa: E402


def run_self_evolution(
    num_generations: int = 10, max_time_hours: float = 1.0, start_server: bool = True, server_port: int = 8504
):
    """
    Run self-evolution.

    [NOTE] 本模块为模拟环境，非真实联邦训练。精度等指标来自合成数据，
           不能作为真实联邦学习能力的证据。

    Args:
        num_generations: Number of generations
        max_time_hours: Maximum time in hours
        start_server: Whether to start the dashboard server
        server_port: Dashboard port
    """
    print("=" * 60)
    print("Self-Evolving Federated Learning")
    print("=" * 60)
    print("[NOTE] 本模块为模拟环境，非真实联邦训练")

    # Create self-evolving system
    sef = SelfEvolvingFederatedLearning()

    # Start server if requested
    if start_server:
        import threading

        server_thread = threading.Thread(target=start_dashboard_server, args=(sef, server_port), daemon=True)
        server_thread.start()

        print(f"\n Dashboard available at: http://localhost:{server_port}")
        time.sleep(2)

    # Run evolution
    sef.evolve(num_generations=num_generations, max_time_hours=max_time_hours)

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
    for algo, stats in report["algorithm_statistics"].items():
        print(f"  {algo}: {stats['mean_accuracy']:.4f} (n={stats['count']})")

    if report["best_configuration"]:
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

    dashboard_html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    app = Flask(__name__)
    CORS(app)

    @app.route("/api/evolution/tree")
    def get_evolution_tree():
        tree = sef.get_evolution_tree()
        return jsonify(tree)

    @app.route("/api/evolution/report")
    def get_evolution_report():
        report = sef.generate_evolution_report()
        return jsonify(report)

    @app.route("/api/evolution/progress")
    def get_progress():
        with open(os.path.join(sef.output_dir, "evolution_progress.json"), "r") as f:
            return jsonify(json.load(f))

    @app.route("/")
    def index():
        return render_template_string(dashboard_html)

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Evolving Federated Learning")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--port", type=int, default=8504)

    args = parser.parse_args()

    run_self_evolution(
        num_generations=args.generations,
        max_time_hours=args.hours,
        start_server=not args.no_server,
        server_port=args.port,
    )
