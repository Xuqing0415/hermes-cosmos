"""
Recommendation Engine for Meta-FL Controller

Selects the best federated learning algorithm based on task characteristics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmRecommender:
    """
    Recommends federated learning algorithms based on task meta-features.
    
    Uses performance predictions and multi-objective optimization to select
    the best algorithm for a given task.
    """
    
    def __init__(self, predictor=None):
        """
        Initialize recommender.
        
        Args:
            predictor: PerformancePredictor instance
        """
        from .performance_predictor import PerformancePredictor
        
        self.predictor = predictor if predictor else PerformancePredictor()
        self.algorithm_descriptions = {
            'fedavg': {
                'name': 'FedAvg',
                'description': 'Standard federated averaging. Good for IID data.',
                'strengths': ['Simple', 'Low overhead', 'Good for IID'],
                'weaknesses': ['Poor on Non-IID', 'Vulnerable to attacks']
            },
            'ditto': {
                'name': 'Ditto',
                'description': 'Personalized FL with proximal term. Handles high Non-IID.',
                'strengths': ['Handles Non-IID', 'Personalized models', 'Fair'],
                'weaknesses': ['Higher communication', 'Requires tuning']
            },
            'fedrep': {
                'name': 'FedRep',
                'description': 'Representation learning approach. Low communication.',
                'strengths': ['Low communication', 'Heterogeneous models'],
                'weaknesses': ['Less personalization', 'Complex']
            },
            'krum': {
                'name': 'Krum',
                'description': 'Byzantine-robust aggregation. Defends against attacks.',
                'strengths': ['Robust to attacks', 'Good security'],
                'weaknesses': ['Higher computation', 'Slower convergence']
            },
            'trimmed_mean': {
                'name': 'Trimmed Mean',
                'description': 'Robust aggregation by trimming outliers.',
                'strengths': ['Simple robust', 'Good for moderate attacks'],
                'weaknesses': ['Loss of information', 'Requires tuning']
            }
        }
    
    def recommend(self, features: Dict[str, float], 
                 objectives: List[str] = None, weights: Dict[str, float] = None
                 ) -> Dict[str, Any]:
        """
        Recommend the best algorithm for the given task.
        
        Args:
            features: Task meta-features
            objectives: List of objectives to optimize (default: ['accuracy', 'communication'])
            weights: Weights for each objective
        
        Returns:
            Recommendation dictionary with algorithm, confidence, and explanation
        """
        if objectives is None:
            objectives = ['accuracy', 'communication']
        
        if weights is None:
            weights = {obj: 1.0 for obj in objectives}
        
        # Get performance predictions
        predictions = self.predictor.predict(features)
        
        # Compute scores for each algorithm
        scores = {}
        
        for algorithm, preds in predictions.items():
            score = 0.0
            total_weight = 0.0
            
            for obj in objectives:
                if obj == 'accuracy':
                    # Higher is better
                    score += weights.get(obj, 1.0) * preds.get(obj, 0)
                    total_weight += weights.get(obj, 1.0)
                elif obj == 'communication':
                    # Lower is better
                    score += weights.get(obj, 1.0) * (2.0 - preds.get(obj, 1))
                    total_weight += weights.get(obj, 1.0)
                elif obj == 'rounds':
                    # Lower is better
                    score += weights.get(obj, 1.0) * (100 - preds.get(obj, 50)) / 100
                    total_weight += weights.get(obj, 1.0)
            
            if total_weight > 0:
                score /= total_weight
            
            scores[algorithm] = score
        
        # Sort algorithms by score
        sorted_algorithms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_algorithm = sorted_algorithms[0][0]
        best_score = sorted_algorithms[0][1]
        
        # Generate explanation
        explanation = self._generate_explanation(best_algorithm, features, predictions)
        
        # Get confidence based on score difference
        if len(sorted_algorithms) > 1:
            confidence = (best_score - sorted_algorithms[1][1]) * 100
        else:
            confidence = 100.0
        
        return {
            'recommended_algorithm': best_algorithm,
            'algorithm_name': self.algorithm_descriptions[best_algorithm]['name'],
            'score': best_score,
            'confidence': min(100.0, max(0.0, confidence)),
            'explanation': explanation,
            'predictions': predictions,
            'sorted_algorithms': [alg for alg, _ in sorted_algorithms],
            'algorithm_info': self.algorithm_descriptions[best_algorithm]
        }
    
    def _generate_explanation(self, algorithm: str, features: Dict[str, float],
                            predictions: Dict[str, Dict[str, float]]) -> str:
        """
        Generate natural language explanation for recommendation.
        
        Args:
            algorithm: Recommended algorithm
            features: Task meta-features
            predictions: Performance predictions
        
        Returns:
            Natural language explanation
        """
        explanations = []
        
        # Non-IID level analysis
        non_iid_score = features.get('non_iid_gini', 0)
        
        if non_iid_score > 0.5:
            if algorithm in ['ditto', 'fedrep']:
                explanations.append(f"Non-IID ({non_iid_score:.2f})")
            else:
                explanations.append(f"Non-IID ({non_iid_score:.2f})")
        else:
            if algorithm == 'fedavg':
                explanations.append(f" (Non-IID={non_iid_score:.2f})FedAvg")
        
        # Client count
        num_clients = features.get('num_clients', 10)
        if num_clients > 30:
            if algorithm == 'fedrep':
                explanations.append(f" ({int(num_clients)})FedRep")
        
        # Bandwidth consideration
        bandwidth = features.get('avg_bandwidth_mbps', 10)
        if bandwidth < 10:
            if algorithm == 'fedrep':
                explanations.append(f" ({bandwidth} Mbps)FedRep")
        
        # Algorithm-specific strengths
        info = self.algorithm_descriptions[algorithm]
        explanations.append(f"{info['name']}: {', '.join(info['strengths'][:2])}")
        
        # Performance comparison
        best_acc = predictions[algorithm]['accuracy']
        other_algs = [a for a in predictions if a != algorithm]
        
        if other_algs:
            best_other_acc = max(predictions[a]['accuracy'] for a in other_algs)
            if best_acc > best_other_acc + 0.05:
                explanations.append(f" ({best_acc:.2f}) ")
        
        return ' '.join(explanations)
    
    def get_pareto_frontier(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get Pareto-optimal algorithms.
        
        Args:
            features: Task meta-features
        
        Returns:
            List of Pareto-optimal algorithms with their metrics
        """
        predictions = self.predictor.predict(features)
        
        # Simple Pareto frontier calculation
        algorithms = []
        
        for algorithm, preds in predictions.items():
            algorithms.append({
                'algorithm': algorithm,
                'accuracy': preds['accuracy'],
                'rounds': preds['rounds'],
                'communication': preds['communication']
            })
        
        # Sort by accuracy, then communication
        algorithms.sort(key=lambda x: (-x['accuracy'], x['communication']))
        
        # Return top 3 as "Pareto frontier" for simplicity
        return algorithms[:3]
    
    def analyze_task(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Analyze task characteristics and provide recommendations.
        
        Args:
            features: Task meta-features
        
        Returns:
            Task analysis dictionary
        """
        analysis = {}
        
        # Non-IID analysis
        non_iid_score = features.get('non_iid_gini', 0)
        if non_iid_score < 0.3:
            analysis['non_iid_level'] = ''
            analysis['non_iid_recommendation'] = 'FedAvg'
        elif non_iid_score < 0.6:
            analysis['non_iid_level'] = ''
            analysis['non_iid_recommendation'] = 'DittoFedRep'
        else:
            analysis['non_iid_level'] = ''
            analysis['non_iid_recommendation'] = 'Ditto'
        
        # Client count analysis
        num_clients = features.get('num_clients', 10)
        if num_clients < 10:
            analysis['client_scale'] = ''
        elif num_clients < 30:
            analysis['client_scale'] = ''
        else:
            analysis['client_scale'] = ''
        
        # Resource analysis
        bandwidth = features.get('avg_bandwidth_mbps', 10)
        if bandwidth < 5:
            analysis['network_status'] = ''
            analysis['network_recommendation'] = 'FedRep'
        elif bandwidth < 50:
            analysis['network_status'] = ''
        else:
            analysis['network_status'] = ''
        
        return analysis
    
    def compare_algorithms(self, features: Dict[str, float], 
                          algorithms: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple algorithms on the same task.
        
        Args:
            features: Task meta-features
            algorithms: List of algorithms to compare
        
        Returns:
            Comparison dictionary
        """
        predictions = self.predictor.predict(features)
        
        if algorithms:
            predictions = {k: v for k, v in predictions.items() if k in algorithms}
        
        comparison = {}
        
        for algorithm, preds in predictions.items():
            comparison[algorithm] = {
                'accuracy': preds['accuracy'],
                'rounds': preds['rounds'],
                'communication': preds['communication'],
                'description': self.algorithm_descriptions[algorithm]['description']
            }
        
        return comparison


if __name__ == "__main__":
    print("=== Testing Algorithm Recommender ===\n")
    
    from .feature_extractor import generate_synthetic_task
    
    # Create recommender
    recommender = AlgorithmRecommender()
    
    # Test with a high Non-IID task
    print("Test 1: High Non-IID Task")
    features = generate_synthetic_task(n_clients=20, non_iid_level=0.8)
    
    recommendation = recommender.recommend(features)
    print(f": {recommendation['algorithm_name']}")
    print(f": {recommendation['confidence']:.1f}%")
    print(f": {recommendation['explanation']}")
    
    print("\n" + "-" * 50)
    
    # Test with a low Non-IID task
    print("\nTest 2: Low Non-IID Task")
    features2 = generate_synthetic_task(n_clients=10, non_iid_level=0.2)
    
    recommendation2 = recommender.recommend(features2)
    print(f": {recommendation2['algorithm_name']}")
    print(f": {recommendation2['confidence']:.1f}%")
    print(f": {recommendation2['explanation']}")
    
    # Task analysis
    print("\n" + "-" * 50)
    print("\nTask Analysis:")
    analysis = recommender.analyze_task(features)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Pareto frontier
    print("\nPareto Frontier:")
    frontier = recommender.get_pareto_frontier(features)
    for alg in frontier:
        print(f"  {alg['algorithm']}: Accuracy={alg['accuracy']:.2f}, Comm={alg['communication']:.2f}x")
