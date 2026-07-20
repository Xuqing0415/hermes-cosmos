"""
Similarity Learner Module
Learns similarity weights from cross-domain migration success rates.
Uses a simple gradient-based approach to adjust the PATTERN_TYPE_SIMILARITY matrix.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import copy


@dataclass
class MigrationRecord:
    """Record of a cross-domain migration attempt."""
    source_pattern_type: str
    target_pattern_type: str
    source_domain: str
    target_domain: str
    similarity_used: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pattern_type": self.source_pattern_type,
            "target_pattern_type": self.target_pattern_type,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "similarity_used": round(self.similarity_used, 2),
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SimilarityMatrix:
    """
    Learnable similarity matrix between pattern types.
    Maps (type_a, type_b) -> similarity weight in [0, 1].
    """
    weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get(self, type_a: str, type_b: str) -> float:
        if type_a == type_b:
            return 1.0
        return self.weights.get(type_a, {}).get(type_b, 0.2)

    def set(self, type_a: str, type_b: str, value: float):
        if type_a not in self.weights:
            self.weights[type_a] = {}
        self.weights[type_a][type_b] = max(0.0, min(1.0, value))

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self.weights.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "SimilarityMatrix":
        return cls(weights={k: dict(v) for k, v in data.items()})


LEARNING_RATE = 0.05
SUCCESS_BOOST = 0.08
FAILURE_PENALTY = 0.04


class SimilarityLearner:
    """
    Learns similarity weights between pattern types based on
    cross-domain migration success/failure history.

    Uses a simple gradient approach:
      - On success: boost the similarity weight for the (source, target) pair
      - On failure: slightly decrease the similarity weight
    """

    def __init__(self, storage_path: str = "loop_output/similarity_matrix.json"):
        self._storage_path = storage_path
        self._matrix = SimilarityMatrix()
        self._history: List[MigrationRecord] = []
        self._migration_count = 0

    @property
    def matrix(self) -> SimilarityMatrix:
        return self._matrix

    @property
    def history(self) -> List[MigrationRecord]:
        return list(self._history)

    @property
    def migration_count(self) -> int:
        return self._migration_count

    def record_migration(self, source_type: str, target_type: str,
                          source_domain: str, target_domain: str,
                          similarity: float, success: bool):
        """Record a migration attempt and update the similarity matrix."""
        record = MigrationRecord(
            source_pattern_type=source_type,
            target_pattern_type=target_type,
            source_domain=source_domain,
            target_domain=target_domain,
            similarity_used=similarity,
            success=success,
        )
        self._history.append(record)
        self._migration_count += 1

        # Update the similarity matrix
        self._update_weight(source_type, target_type, success)

    def get_pattern_type_similarity(self) -> Dict[str, Dict[str, float]]:
        """
        Export the learned similarity matrix in the same format as
        PATTERN_TYPE_SIMILARITY in pattern_similarity_engine.py.
        """
        result = {}
        for type_a, neighbors in self._matrix.weights.items():
            result[type_a] = dict(neighbors)
        return result

    def get_success_rate(self, source_type: str, target_type: str) -> float:
        """Get the success rate for migrations between two pattern types."""
        relevant = [
            r for r in self._history
            if r.source_pattern_type == source_type
            and r.target_pattern_type == target_type
        ]
        if not relevant:
            return 0.0
        successes = sum(1 for r in relevant if r.success)
        return successes / len(relevant)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the learner."""
        if not self._history:
            return {"total_migrations": 0, "success_rate": 0.0}

        successes = sum(1 for r in self._history if r.success)
        return {
            "total_migrations": self._migration_count,
            "success_rate": round(successes / len(self._history), 2),
            "matrix_entries": sum(
                len(neighbors) for neighbors in self._matrix.weights.values()
            ),
            "history_size": len(self._history),
        }

    def save(self):
        """Save the similarity matrix and migration history to disk."""
        data = {
            "matrix": self._matrix.to_dict(),
            "history": [r.to_dict() for r in self._history],
            "migration_count": self._migration_count,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        directory = os.path.dirname(self._storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load the similarity matrix and migration history from disk."""
        if not os.path.exists(self._storage_path):
            return

        with open(self._storage_path, "r") as f:
            data = json.load(f)

        self._matrix = SimilarityMatrix.from_dict(data.get("matrix", {}))
        self._history = []
        for r_data in data.get("history", []):
            self._history.append(MigrationRecord(
                source_pattern_type=r_data["source_pattern_type"],
                target_pattern_type=r_data["target_pattern_type"],
                source_domain=r_data["source_domain"],
                target_domain=r_data["target_domain"],
                similarity_used=r_data.get("similarity_used", 0.0),
                success=r_data["success"],
                timestamp=datetime.fromisoformat(
                    r_data.get("timestamp", datetime.now(timezone.utc).isoformat())
                ),
            ))
        self._migration_count = data.get("migration_count", len(self._history))

    def _update_weight(self, source_type: str, target_type: str, success: bool):
        """Update the similarity weight based on migration outcome."""
        current = self._matrix.get(source_type, target_type)
        delta = SUCCESS_BOOST if success else -FAILURE_PENALTY
        new_weight = current + delta * LEARNING_RATE
        self._matrix.set(source_type, target_type, new_weight)
        self._matrix.set(target_type, source_type, new_weight)

    def apply_to_initial_matrix(self, initial_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Merge the learned weights into the initial hardcoded matrix.
        Learned weights take precedence (overwrite) for known pairs.
        New pairs discovered by the learner are added.
        """
        result = copy.deepcopy(initial_matrix)
        for type_a, neighbors in self._matrix.weights.items():
            if type_a not in result:
                result[type_a] = {}
            for type_b, weight in neighbors.items():
                result[type_a][type_b] = weight
        return result