"""
Meta-FL Controller Experiment Runner

Runs meta-learning based algorithm selection for federated learning.
"""

import os
import sys
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_fl import TaskFeatureExtractor, PerformancePredictor, AlgorithmRecommender


def run_meta_experiment(client_data=None, resources=None, objectives=None):
    """
    Run complete meta-learning experiment.
    
    Args:
        client_data: Optional client data for feature extraction
        resources: Optional resource constraints
        objectives: Optimization objectives
    
    Returns:
        Recommendation results
    """
    print("=" * 60)
    print("Meta-FL Controller: Automatic Algorithm Selection")
    print("=" * 60)
    
    # Step 1: Extract meta-features
    print("\nStep 1: Extracting Task Meta-Features")
    
    if client_data is None:
        from meta_fl.feature_extractor import generate_synthetic_task
        
        # Generate synthetic task for testing
        features = generate_synthetic_task(n_clients=20, non_iid_level=0.6)
        print("  Generated synthetic task features")
    else:
        extractor = TaskFeatureExtractor()
        features = extractor.extract(client_data, resources)
        print("  Extracted features from client data")
    
    # Print key features
    print("\nKey Task Features:")
    print(f"  Clients: {int(features.get('num_clients', 0))}")
    print(f"  Non-IID Level: {features.get('non_iid_gini', 0):.2f}")
    print(f"  Classes: {int(features.get('num_classes', 0))}")
    print(f"  Bandwidth: {features.get('avg_bandwidth_mbps', 0):.1f} Mbps")
    
    # Step 2: Analyze task
    print("\nStep 2: Task Analysis")
    recommender = AlgorithmRecommender()
    analysis = recommender.analyze_task(features)
    
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Step 3: Generate recommendation
    print("\nStep 3: Generating Recommendation")
    
    if objectives is None:
        objectives = ['accuracy', 'communication']
    
    recommendation = recommender.recommend(features, objectives=objectives)
    
    print(f"\nRecommended Algorithm: {recommendation['algorithm_name']}")
    print(f"Confidence: {recommendation['confidence']:.1f}%")
    print(f"\nExplanation:")
    print(f"  {recommendation['explanation']}")
    
    # Step 4: Show comparison
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    
    comparison = recommender.compare_algorithms(features)
    
    print(f"\n{'Algorithm':<15} {'Accuracy':<10} {'Rounds':<8} {'Communication':<15}")
    print("-" * 50)
    
    for algorithm, metrics in comparison.items():
        print(f"{algorithm:<15} {metrics['accuracy']:<10.4f} {metrics['rounds']:<8.1f} {metrics['communication']:<15.2f}x")
    
    # Step 5: Feature importance
    print("\n" + "=" * 60)
    print("Feature Importance for Recommendation")
    print("=" * 60)
    
    importances = recommender.predictor.get_feature_importance()
    print("\nTop 5 Features:")
    for feature, importance in importances[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    return recommendation


def start_recommendation_server(port: int = 8503):
    """Start the Meta-FL recommendation server."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/recommend', methods=['POST'])
    def recommend():
        """Get algorithm recommendation."""
        data = request.json
        
        if 'features' in data:
            features = data['features']
        else:
            # Generate synthetic features
            from meta_fl.feature_extractor import generate_synthetic_task
            features = generate_synthetic_task(
                n_clients=data.get('num_clients', 10),
                non_iid_level=data.get('non_iid_level', 0.5)
            )
        
        objectives = data.get('objectives', ['accuracy', 'communication'])
        
        recommender = AlgorithmRecommender()
        recommendation = recommender.recommend(features, objectives=objectives)
        
        return jsonify(recommendation)
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """Analyze task characteristics."""
        data = request.json
        
        if 'features' in data:
            features = data['features']
        else:
            from meta_fl.feature_extractor import generate_synthetic_task
            features = generate_synthetic_task(
                n_clients=data.get('num_clients', 10),
                non_iid_level=data.get('non_iid_level', 0.5)
            )
        
        recommender = AlgorithmRecommender()
        analysis = recommender.analyze_task(features)
        frontier = recommender.get_pareto_frontier(features)
        
        return jsonify({
            'analysis': analysis,
            'pareto_frontier': frontier,
            'features': features
        })
    
    @app.route('/')
    def index():
        return """
        <h1>Meta-FL Controller API</h1>
        <p>POST to /api/recommend for algorithm recommendation</p>
        <p>POST to /api/analyze for task analysis</p>
        """
    
    print(f"Meta-FL Controller API running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Meta-FL Controller')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'server'])
    parser.add_argument('--port', type=int, default=8503)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--non-iid', type=float, default=0.6)
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        start_recommendation_server(args.port)
    else:
        run_meta_experiment()
