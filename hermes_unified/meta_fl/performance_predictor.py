"""
Performance Predictor for Meta-FL Controller

Predicts algorithm performance based on task meta-features.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformancePredictor:
    """
    Predicts algorithm performance using meta-learning.
    
    Trains a regression model on historical experiment data to predict:
    - Final accuracy
    - Convergence rounds
    - Communication cost
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize performance predictor.
        
        Args:
            model_path: Path to saved model
        """
        self.models = {
            'accuracy': RandomForestRegressor(n_estimators=100, random_state=42),
            'rounds': RandomForestRegressor(n_estimators=100, random_state=42),
            'communication': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        self.feature_names = None
        self.trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def generate_synthetic_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic experiment data for training.
        
        Args:
            n_samples: Number of synthetic experiments
        
        Returns:
            (X, y, metadata)
        """
        from .feature_extractor import generate_synthetic_task
        
        X = []
        y_accuracy = []
        y_rounds = []
        y_communication = []
        algorithms = []
        
        algorithms_list = ['fedavg', 'ditto', 'fedrep', 'krum', 'trimmed_mean']
        
        for _ in range(n_samples):
            # Generate random task features
            n_clients = np.random.randint(5, 50)
            non_iid_level = np.random.uniform(0, 1)
            
            features = generate_synthetic_task(
                n_clients=n_clients,
                non_iid_level=non_iid_level
            )
            
            X.append(list(features.values()))
            
            # Pick random algorithm
            algorithm = np.random.choice(algorithms_list)
            algorithms.append(algorithm)
            
            # Simulate performance based on algorithm and features
            base_accuracy = 0.6 + non_iid_level * 0.3  # Lower accuracy for more Non-IID
            
            # Algorithm-specific adjustments
            if algorithm == 'ditto':
                accuracy = base_accuracy + min(non_iid_level * 0.15, 0.1)
                rounds = max(10, int(50 - n_clients * 0.5))
                communication = 1.0
            elif algorithm == 'fedrep':
                accuracy = base_accuracy + min(non_iid_level * 0.1, 0.08)
                rounds = max(8, int(40 - n_clients * 0.4))
                communication = 0.5  # Lower communication
            elif algorithm == 'krum':
                accuracy = base_accuracy + 0.05  # Robust to attacks
                rounds = max(12, int(55 - n_clients * 0.4))
                communication = 1.2
            elif algorithm == 'trimmed_mean':
                accuracy = base_accuracy + 0.03
                rounds = max(10, int(45 - n_clients * 0.3))
                communication = 1.1
            else:  # fedavg
                accuracy = base_accuracy
                rounds = max(15, int(60 - n_clients * 0.5))
                communication = 1.0
            
            # Add noise
            accuracy += np.random.normal(0, 0.03)
            rounds += np.random.randint(-5, 5)
            
            y_accuracy.append(min(max(0.3, accuracy), 0.95))
            y_rounds.append(max(5, rounds))
            y_communication.append(max(0.3, communication + np.random.normal(0, 0.1)))
        
        X = np.array(X)
        y = {
            'accuracy': np.array(y_accuracy),
            'rounds': np.array(y_rounds),
            'communication': np.array(y_communication)
        }
        
        metadata = {
            'feature_names': list(features.keys()),
            'algorithms': algorithms
        }
        
        return X, y, metadata
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray], feature_names: List[str]):
        """
        Train performance prediction models.
        
        Args:
            X: Feature matrix
            y: Dictionary of target variables
            feature_names: List of feature names
        """
        logger.info(f"Training performance predictor on {len(X)} samples")
        
        self.feature_names = feature_names
        
        for target, model in self.models.items():
            if target in y:
                X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"  {target}: MAE={mae:.4f}, R2={r2:.4f}")
        
        self.trained = True
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Predict performance for all algorithms.
        
        Args:
            features: Task meta-features
        
        Returns:
            Dictionary of predictions per algorithm
        """
        if not self.trained:
            # Train on synthetic data if not trained
            X, y, metadata = self.generate_synthetic_dataset()
            self.train(X, y, metadata['feature_names'])
        
        # Convert features to array
        X = np.array([list(features.values())])
        
        algorithms = ['fedavg', 'ditto', 'fedrep', 'krum', 'trimmed_mean']
        predictions = {}
        
        for algorithm in algorithms:
            # For simplicity, we use the same model but adjust predictions based on algorithm
            # In a real system, you'd have separate models per algorithm
            
            base_preds = {
                target: float(model.predict(X)[0])
                for target, model in self.models.items()
            }
            
            # Apply algorithm-specific adjustments
            if algorithm == 'ditto':
                base_preds['accuracy'] *= 1.05
                base_preds['rounds'] *= 0.9
                base_preds['communication'] *= 1.0
            elif algorithm == 'fedrep':
                base_preds['accuracy'] *= 1.03
                base_preds['rounds'] *= 0.85
                base_preds['communication'] *= 0.5
            elif algorithm == 'krum':
                base_preds['accuracy'] *= 1.02
                base_preds['rounds'] *= 1.05
                base_preds['communication'] *= 1.2
            elif algorithm == 'trimmed_mean':
                base_preds['accuracy'] *= 1.01
                base_preds['rounds'] *= 0.95
                base_preds['communication'] *= 1.1
            
            predictions[algorithm] = base_preds
        
        return predictions
    
    def get_feature_importance(self, target: str = 'accuracy') -> List[Tuple[str, float]]:
        """
        Get feature importance for a specific target.
        
        Args:
            target: Target variable ('accuracy', 'rounds', 'communication')
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.trained or self.feature_names is None:
            return []
        
        model = self.models.get(target)
        if model is None:
            return []
        
        importances = model.feature_importances_
        pairs = list(zip(self.feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pairs
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        import pickle
        
        data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'trained': self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.feature_names = data['feature_names']
        self.trained = data['trained']
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    print("=== Testing Performance Predictor ===\n")
    
    # Create predictor
    predictor = PerformancePredictor()
    
    # Generate synthetic dataset and train
    X, y, metadata = predictor.generate_synthetic_dataset(n_samples=500)
    predictor.train(X, y, metadata['feature_names'])
    
    # Test prediction with a new task
    from .feature_extractor import generate_synthetic_task
    
    features = generate_synthetic_task(n_clients=15, non_iid_level=0.6)
    predictions = predictor.predict(features)
    
    print("Predictions for sample task:")
    print("-" * 50)
    
    for algorithm, preds in predictions.items():
        print(f"\n{algorithm}:")
        print(f"  Accuracy: {preds['accuracy']:.4f}")
        print(f"  Rounds: {preds['rounds']:.1f}")
        print(f"  Communication: {preds['communication']:.2f}x baseline")
    
    # Feature importance
    print("\nFeature Importance for Accuracy:")
    for feature, importance in predictor.get_feature_importance()[:5]:
        print(f"  {feature}: {importance:.4f}")
