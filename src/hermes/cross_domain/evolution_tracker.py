from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3
import os
import json
import hashlib


@dataclass
class EvolutionSnapshot:
    id: int
    snapshot_index: int
    success_rate: float
    cross_domain_success: float
    pattern_coverage: float
    knowledge_graph_hash: str
    total_patterns: int
    total_relationships: int
    policy_mutation_rate: float
    policy_exploration_factor: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "snapshot_index": self.snapshot_index,
            "success_rate": self.success_rate,
            "cross_domain_success": self.cross_domain_success,
            "pattern_coverage": self.pattern_coverage,
            "knowledge_graph_hash": self.knowledge_graph_hash,
            "total_patterns": self.total_patterns,
            "total_relationships": self.total_relationships,
            "policy_mutation_rate": self.policy_mutation_rate,
            "policy_exploration_factor": self.policy_exploration_factor,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS evolution_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_index INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    cross_domain_success REAL NOT NULL,
    pattern_coverage REAL NOT NULL,
    knowledge_graph_hash TEXT NOT NULL,
    total_patterns INTEGER NOT NULL,
    total_relationships INTEGER NOT NULL,
    policy_mutation_rate REAL NOT NULL DEFAULT 0.1,
    policy_exploration_factor REAL NOT NULL DEFAULT 0.15,
    timestamp TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_snapshot_index ON evolution_snapshots(snapshot_index);
"""


def compute_graph_hash(knowledge_amalgamator) -> str:
    """Compute a hash of the knowledge graph for change detection."""
    graph = knowledge_amalgamator.get_graph() if hasattr(knowledge_amalgamator, 'get_graph') else None
    if graph is None:
        return "empty"

    node_data = [(n.id, n.pattern_type.value, sorted(n.domains),
                  round(n.confidence, 2), n.occurrences) for n in graph.nodes]
    edge_data = [(e.source_id, e.target_id, round(e.similarity, 2),
                  e.relationship_type) for e in graph.edges]

    combined = json.dumps({"nodes": sorted(node_data), "edges": sorted(edge_data)},
                          sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class EvolutionTracker:
    def __init__(self, db_path: str = "loop_output/evolution_tracker.db"):
        self._db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        directory = os.path.dirname(self._db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(DB_SCHEMA)
        finally:
            conn.close()

    def record_snapshot(self, success_rate: float, cross_domain_success: float,
                        pattern_coverage: float, knowledge_amalgamator=None,
                        policy=None, metadata: Optional[Dict[str, Any]] = None) -> EvolutionSnapshot:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(snapshot_index), 0) FROM evolution_snapshots"
            )
            last_index = cursor.fetchone()[0]
            snapshot_index = last_index + 1

            kg_hash = compute_graph_hash(knowledge_amalgamator) if knowledge_amalgamator else "unknown"

            total_patterns = 0
            total_relationships = 0
            if knowledge_amalgamator:
                total_patterns = knowledge_amalgamator.get_pattern_count() if hasattr(knowledge_amalgamator, 'get_pattern_count') else 0
                total_relationships = knowledge_amalgamator.get_relationship_count() if hasattr(knowledge_amalgamator, 'get_relationship_count') else 0

            mutation_rate = 0.1
            exploration_factor = 0.15
            if policy:
                mutation_rate = getattr(policy, 'mutation_rate', 0.1)
                exploration_factor = getattr(policy, 'exploration_factor', 0.15)

            now = datetime.now(timezone.utc)
            meta_json = json.dumps(metadata or {})

            conn.execute(
                """INSERT INTO evolution_snapshots
                   (snapshot_index, success_rate, cross_domain_success, pattern_coverage,
                    knowledge_graph_hash, total_patterns, total_relationships,
                    policy_mutation_rate, policy_exploration_factor, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (snapshot_index, success_rate, cross_domain_success, pattern_coverage,
                 kg_hash, total_patterns, total_relationships,
                 mutation_rate, exploration_factor, now.isoformat(), meta_json)
            )
            conn.commit()

            return EvolutionSnapshot(
                id=snapshot_index,
                snapshot_index=snapshot_index,
                success_rate=success_rate,
                cross_domain_success=cross_domain_success,
                pattern_coverage=pattern_coverage,
                knowledge_graph_hash=kg_hash,
                total_patterns=total_patterns,
                total_relationships=total_relationships,
                policy_mutation_rate=mutation_rate,
                policy_exploration_factor=exploration_factor,
                timestamp=now,
                metadata=metadata or {},
            )
        finally:
            conn.close()

    def get_snapshots(self, limit: int = 100, offset: int = 0) -> List[EvolutionSnapshot]:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """SELECT id, snapshot_index, success_rate, cross_domain_success,
                          pattern_coverage, knowledge_graph_hash, total_patterns,
                          total_relationships, policy_mutation_rate,
                          policy_exploration_factor, timestamp, metadata
                   FROM evolution_snapshots
                   ORDER BY snapshot_index DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            )
            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(EvolutionSnapshot(
                    id=row[0],
                    snapshot_index=row[1],
                    success_rate=row[2],
                    cross_domain_success=row[3],
                    pattern_coverage=row[4],
                    knowledge_graph_hash=row[5],
                    total_patterns=row[6],
                    total_relationships=row[7],
                    policy_mutation_rate=row[8],
                    policy_exploration_factor=row[9],
                    timestamp=datetime.fromisoformat(row[10]),
                    metadata=json.loads(row[11]) if row[11] else {},
                ))
            return snapshots
        finally:
            conn.close()

    def get_recent_snapshots(self, n: int = 10) -> List[EvolutionSnapshot]:
        snapshots = self.get_snapshots(limit=n)
        snapshots.reverse()
        return snapshots

    def count_snapshots(self) -> int:
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM evolution_snapshots")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def clear(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("DELETE FROM evolution_snapshots")
            conn.commit()
        finally:
            conn.close()