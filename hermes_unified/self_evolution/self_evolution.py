"""
Self-Evolving Federated Learning Core

Main loop for autonomous experimentation and algorithm improvement.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import time
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfEvolvingFederatedLearning:
    """
    Self-evolving FL system that automatically experiments and improves.
    
    Features:
    - Random parameter space exploration
    - Bayesian optimization/UCB for smart selection
    - Online learning for performance prediction
    - Automatic deployment of best configurations
    """
    
    def __init__(self, db_path: str = "self_evolution_db.json",
                 output_dir: str = "self_evolution_results"):
        """
        Initialize self-evolving FL system.
        
        Args:
            db_path: Path to store experiment history
            output_dir: Directory to save results
        """
        self.db_path = db_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing experiment database
        self.experiment_db = self._load_database()
        
        # Components
        from meta_fl import AlgorithmRecommender
        from meta_fl.feature_extractor import generate_synthetic_task
        
        self.recommender = AlgorithmRecommender()
        
        # Exploration parameters
        self.ucb_c = 1.0  # UCB exploration constant
        self.epsilon = 0.1  # Random exploration probability
        
        # Evolution stats
        self.generation = 0
        self.milestones = []
        self.best_performance = 0.0
        
        # Current best configuration
        self.best_config = None
        
    def _load_database(self) -> List[Dict]:
        """Load experiment database from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_database(self):
        """Save experiment database to disk."""
        with open(self.db_path, 'w') as f:
            json.dump(self.experiment_db, f, indent=2)
    
    def generate_random_config(self) -> Dict[str, Any]:
        """
        Generate random experiment configuration.
        
        Returns:
            Random configuration dictionary
        """
        algorithms = ['fedavg', 'ditto', 'fedrep', 'krum', 'trimmed_mean']
        
        config = {
            'num_clients': np.random.randint(5, 50),
            'non_iid_level': np.random.uniform(0.1, 0.95),
            'algorithm': np.random.choice(algorithms),
            'learning_rate': np.random.choice([0.01, 0.05, 0.1]),
            'num_rounds': np.random.randint(10, 50),
            'bandwidth': np.random.uniform(1, 100),
            'compute_power': np.random.uniform(0.3, 2.0)
        }
        
        return config
    
    def select_config_ucb(self) -> Dict[str, Any]:
        """
        Select next configuration using Upper Confidence Bound.
        
        Balances exploration and exploitation.
        
        Returns:
            Selected configuration
        """
        if len(self.experiment_db) < 10 or np.random.random() < self.epsilon:
            # Random exploration
            return self.generate_random_config()
        
        # UCB selection based on past results
        config_groups = {}
        
        for exp in self.experiment_db:
            key = (exp['num_clients'], exp['algorithm'])
            if key not in config_groups:
                config_groups[key] = {'results': [], 'count': 0}
            config_groups[key]['results'].append(exp['accuracy'])
            config_groups[key]['count'] += 1
        
        # Compute UCB scores
        best_score = -1
        best_key = None
        
        for key, group in config_groups.items():
            mean_acc = np.mean(group['results'])
            count = group['count']
            ucb_score = mean_acc + self.ucb_c * np.sqrt(np.log(len(self.experiment_db)) / count)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_key = key
        
        # Generate similar configuration
        if best_key:
            num_clients, algorithm = best_key
            config = self.generate_random_config()
            config['num_clients'] = num_clients
            config['algorithm'] = algorithm
            return config
        
        return self.generate_random_config()
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Experiment configuration
        
        Returns:
            Experiment results
        """
        logger.info(f"Running experiment with config: {config}")
        
        # Simulate experiment (in real system, this would run actual FL)
        # For demo purposes, we'll simulate based on configuration
        
        from meta_fl.feature_extractor import generate_synthetic_task
        
        features = generate_synthetic_task(
            n_clients=config['num_clients'],
            non_iid_level=config['non_iid_level']
        )
        
        # Get predictions
        predictions = self.recommender.predictor.predict(features)
        algorithm_preds = predictions[config['algorithm']]
        
        # Simulate "actual" performance with some noise
        actual_accuracy = algorithm_preds['accuracy'] + np.random.normal(0, 0.03)
        actual_rounds = algorithm_preds['rounds'] + np.random.randint(-5, 5)
        
        result = {
            'experiment_id': len(self.experiment_db),
            'timestamp': datetime.now().isoformat(),
            'generation': self.generation,
            'config': config,
            'meta_features': features,
            'predicted_accuracy': algorithm_preds['accuracy'],
            'predicted_rounds': algorithm_preds['rounds'],
            'predicted_communication': algorithm_preds['communication'],
            'actual_accuracy': float(actual_accuracy),
            'actual_rounds': int(actual_rounds),
            'prediction_error': float(abs(actual_accuracy - algorithm_preds['accuracy']))
        }
        
        # Add to database
        self.experiment_db.append(result)
        self._save_database()
        
        # Check if this is a milestone
        if actual_accuracy > self.best_performance + 0.02:
            self.best_performance = actual_accuracy
            self.best_config = config
            self.milestones.append(result)
            logger.info(f"🎉 New milestone! Accuracy: {actual_accuracy:.4f}")
        
        return result
    
    def update_predictor_online(self, result: Dict[str, Any]):
        """
        Update performance predictor with new experiment data.
        
        Args:
            result: Experiment result
        """
        # In a real system, this would perform online learning
        # For demo, we'll just log
        pass
    
    def auto_deploy_best_config(self):
        """
        Auto-deploy the best configuration found.
        
        Simulates GitOps deployment update.
        """
        if self.best_config is None:
            return
        
        logger.info(f"🔧 Auto-deploying best configuration: {self.best_config}")
        
        # Simulate deployment update
        deployment_config = {
            'best_algorithm': self.best_config['algorithm'],
            'best_num_clients': self.best_config['num_clients'],
            'best_learning_rate': self.best_config.get('learning_rate', 0.01),
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.best_performance
        }
        
        # Save deployment config
        with open(os.path.join(self.output_dir, 'best_deployment.json'), 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        logger.info("✅ Deployment configuration updated")
    
    def evolve(self, num_generations: int = 10, max_time_hours: float = 1.0):
        """
        Run the self-evolution loop.
        
        Args:
            num_generations: Number of generations to run
            max_time_hours: Maximum time in hours
        """
        logger.info("=" * 60)
        logger.info("Starting Self-Evolving Federated Learning")
        logger.info("=" * 60)
        
        start_time = time.time()
        max_time_seconds = max_time_hours * 3600
        
        for gen in range(num_generations):
            self.generation = gen
            
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                logger.warning("⏰ Time limit reached, stopping evolution")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {gen + 1}/{num_generations}")
            logger.info(f"{'='*60}")
            
            # Select configuration
            config = self.select_config_ucb()
            
            # Run experiment
            result = self.run_single_experiment(config)
            
            # Update online predictor
            self.update_predictor_online(result)
            
            # Auto-deploy if milestone
            if result['experiment_id'] == len(self.experiment_db) - 1:
                if result in self.milestones:
                    self.auto_deploy_best_config()
            
            # Summary
            logger.info(f"\nExperiment {result['experiment_id']} Summary:")
            logger.info(f"  Algorithm: {config['algorithm']}")
            logger.info(f"  Predicted Accuracy: {result['predicted_accuracy']:.4f}")
            logger.info(f"  Actual Accuracy: {result['actual_accuracy']:.4f}")
            logger.info(f"  Error: {result['prediction_error']:.4f}")
            
            # Save intermediate progress
            self.save_progress()
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("Evolution Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total Experiments: {len(self.experiment_db)}")
        logger.info(f"Milestones: {len(self.milestones)}")
        logger.info(f"Best Accuracy: {self.best_performance:.4f}")
        
        if self.best_config:
            logger.info(f"Best Config: {self.best_config}")
        
        # Auto-deploy final best
        self.auto_deploy_best_config()
        
        # Save final results
        self.save_progress()
    
    def save_progress(self):
        """Save evolution progress to disk."""
        progress = {
            'generation': self.generation,
            'best_performance': float(self.best_performance),
            'best_config': self.best_config,
            'milestones': self.milestones,
            'total_experiments': len(self.experiment_db),
            'experiment_db': self.experiment_db[-100:],  # Last 100 experiments
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'evolution_progress.json'), 'w') as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"📊 Progress saved. Experiments: {len(self.experiment_db)}")
    
    def get_evolution_tree(self) -> List[Dict]:
        """
        Get evolution tree data for visualization.
        
        Returns:
            Evolution tree data
        """
        tree_data = []
        
        for exp in self.experiment_db[-50:]:  # Last 50 experiments
            node = {
                'experiment_id': exp['experiment_id'],
                'generation': exp['generation'],
                'algorithm': exp['config']['algorithm'],
                'predicted_accuracy': exp['predicted_accuracy'],
                'actual_accuracy': exp['actual_accuracy'],
                'is_milestone': exp in self.milestones,
                'config_summary': {
                    'num_clients': exp['config']['num_clients'],
                    'non_iid_level': exp['config']['non_iid_level'],
                    'bandwidth': exp['config']['bandwidth']
                }
            }
            tree_data.append(node)
        
        return tree_data
    
    def generate_evolution_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evolution report.
        
        Returns:
            Report dictionary
        """
        if not self.experiment_db:
            return {'error': 'No experiments run yet'}
        
        accuracies = [exp['actual_accuracy'] for exp in self.experiment_db]
        algorithms_used = [exp['config']['algorithm'] for exp in self.experiment_db]
        
        # Algorithm performance
        from collections import defaultdict
        algo_perf = defaultdict(list)
        for exp in self.experiment_db:
            algo_perf[exp['config']['algorithm']].append(exp['actual_accuracy'])
        
        algo_stats = {}
        for algo, perfs in algo_perf.items():
            algo_stats[algo] = {
                'mean_accuracy': float(np.mean(perfs)),
                'std_accuracy': float(np.std(perfs)),
                'count': len(perfs)
            }
        
        return {
            'summary': {
                'total_experiments': len(self.experiment_db),
                'generations': self.generation + 1,
                'best_accuracy': float(self.best_performance),
                'milestones': len(self.milestones)
            },
            'accuracy_trend': accuracies[-20:],  # Last 20
            'algorithm_statistics': algo_stats,
            'best_configuration': self.best_config,
            'milestones': [m['experiment_id'] for m in self.milestones]
        }


if __name__ == "__main__":
    print("=== Testing Self-Evolving FL ===\n")
    
    # Create self-evolving system
    sef = SelfEvolvingFederatedLearning()
    
    # Run evolution for 3 generations (quick demo)
    sef.evolve(num_generations=3, max_time_hours=0.5)
    
    # Get evolution tree
    tree = sef.get_evolution_tree()
    print(f"\nEvolution Tree Nodes: {len(tree)}")
    
    # Generate report
    report = sef.generate_evolution_report()
    print(f"\nEvolution Report:")
    print(f"  Total Experiments: {report['summary']['total_experiments']}")
    print(f"  Best Accuracy: {report['summary']['best_accuracy']:.4f}")
    print(f"  Milestones: {report['summary']['milestones']}")
