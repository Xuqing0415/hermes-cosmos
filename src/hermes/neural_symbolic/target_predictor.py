"""
Neural Proof Target Predictor - Predicts which paths need proof
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Optional, Any, Tuple
import structlog
import joblib
import os

from hermes.neural_symbolic.types import ProofTarget, PathCondition

logger = structlog.get_logger()


class NeuralProofTargetPredictor:
    """
    Predicts which paths are most likely to require proof.
    
    Uses a Random Forest classifier to predict proof difficulty/failure likelihood.
    Features include: path condition complexity, variable types, operation types, etc.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._train_on_synthetic_data()
    
    def _extract_features(self, target: ProofTarget) -> np.ndarray:
        """
        Extract features from a proof target for prediction.
        
        Features:
        - Path condition length
        - Number of variables
        - Number of operations
        - Has division operation
        - Has comparison operation
        - Has logical operation
        - Variable count
        - Risk score (from proof obligation generator)
        """
        condition = target.path_condition.condition
        variables = target.path_condition.variables
        
        features = [
            len(condition),
            len(variables),
            condition.count("("),
            1 if "/" in condition or "div" in condition.lower() else 0,
            1 if ">" in condition or "<" in condition or ">=" in condition or "<=" in condition else 0,
            1 if "and" in condition.lower() or "or" in condition.lower() or "not" in condition.lower() else 0,
            1 if "=" in condition else 0,
            target.risk_score
        ]
        
        return np.array(features)
    
    def _train_on_synthetic_data(self):
        """Train the model on synthetic data"""
        X, y = self._generate_synthetic_dataset()
        
        if len(X) > 0:
            self.model.fit(X, y)
            self.trained = True
            logger.info("Model trained on synthetic data", samples=len(X))
    
    def _generate_synthetic_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        samples = []
        labels = []
        
        for _ in range(500):
            has_div = np.random.rand() > 0.7
            has_compare = np.random.rand() > 0.5
            has_logic = np.random.rand() > 0.5
            condition_len = np.random.randint(10, 200)
            var_count = np.random.randint(1, 10)
            paren_count = np.random.randint(1, 20)
            has_eq = np.random.rand() > 0.5
            risk_score = min(0.3 + has_div * 0.5 + has_eq * 0.2, 1.0)
            
            features = [
                condition_len,
                var_count,
                paren_count,
                int(has_div),
                int(has_compare),
                int(has_logic),
                int(has_eq),
                risk_score
            ]
            
            failure_prob = 0.2 + has_div * 0.5 + risk_score * 0.3
            label = 1 if np.random.rand() < failure_prob else 0
            
            samples.append(features)
            labels.append(label)
        
        return np.array(samples), np.array(labels)
    
    def predict(self, targets: List[ProofTarget]) -> List[Tuple[ProofTarget, float]]:
        """
        Predict which targets are most likely to require proof.
        
        Args:
            targets: List of ProofTarget objects
        
        Returns:
            List of (target, failure_probability) tuples
        """
        if not self.trained or len(targets) == 0:
            return [(t, t.risk_score) for t in targets]
        
        features = np.array([self._extract_features(t) for t in targets])
        probs = self.model.predict_proba(features)[:, 1]
        
        return list(zip(targets, probs))
    
    def predict_top_targets(
        self,
        targets: List[ProofTarget],
        top_n: int = 5
    ) -> List[ProofTarget]:
        """
        Predict the top N targets that are most likely to require proof.
        
        Args:
            targets: List of ProofTarget objects
            top_n: Number of top targets to return
        
        Returns:
            List of top ProofTarget objects sorted by priority
        """
        predictions = self.predict(targets)
        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        for target, prob in sorted_predictions[:top_n]:
            target.risk_score = prob
            target.priority = int(prob * 10)
        
        return [t for t, _ in sorted_predictions[:top_n]]
    
    def save_model(self, path: str):
        """
        Save the trained model to a file.
        
        Args:
            path: Path to save the model
        """
        joblib.dump({
            "model": self.model,
            "encoder": self.encoder,
            "trained": self.trained
        }, path)
        logger.info("Model saved", path=path)
    
    def load_model(self, path: str):
        """
        Load a trained model from a file.
        
        Args:
            path: Path to the model file
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.encoder = data["encoder"]
        self.trained = data["trained"]
        logger.info("Model loaded", path=path)
    
    def update_with_results(
        self,
        targets: List[ProofTarget],
        success: List[bool]
    ):
        """
        Update the model with new proof results.
        
        Args:
            targets: List of ProofTarget objects
            success: List of booleans indicating proof success (True = proven)
        """
        if len(targets) == 0:
            return
        
        X = np.array([self._extract_features(t) for t in targets])
        y = np.array([0 if s else 1 for s in success])
        
        self.model.fit(X, y)
        self.trained = True
        logger.info("Model updated with new results", samples=len(targets))
