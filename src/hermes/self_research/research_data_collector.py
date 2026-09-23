"""Self-Researcher 数据收集层。

论文需要的数据分散在多个既有子系统中，本模块把它们收敛成统一的数据集：

* 演化快照：`EvolutionTracker` 的 SQLite 表
* 策略事件：`loop_output/notifications.log` 的 JSON Lines 日志
* 改进策略：`SelfImprovementPolicy` 的 JSON 策略文件
* 知识图谱：`KnowledgeAmalgamator` 的 JSON 图谱
* 演化阶段：`SelfRepositoryMiner` 对自身 Git 历史的划分

所有数据源都是可选的：文件缺失或损坏时退化为空集合而不是抛错，
这样在没有任何历史产物的环境下也能跑完整条论文生成流水线。
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hermes.cross_domain.evolution_tracker import EvolutionSnapshot, EvolutionTracker
from hermes.cross_domain.knowledge_amalgamator import KnowledgeAmalgamator
from hermes.cross_domain.mental_model_trainer import MentalModelTrainer
from hermes.cross_domain.self_improvement_policy import SelfImprovementPolicy
from hermes.cross_domain.self_repository_miner import SelfRepositoryMiner

DEFAULT_SNAPSHOT_DB = "loop_output/evolution_tracker.db"
DEFAULT_POLICY_PATH = "loop_output/self_improvement_policy.json"
DEFAULT_NOTIFICATION_LOG = "loop_output/notifications.log"
DEFAULT_GRAPH_PATH = "loop_output/knowledge_graph.json"

EXPECTED_GAIN_RE = re.compile(r"预期收益:\s*([0-9.]+)%")

# 事件类型 -> 策略类型，用于把事件日志折算成可比较的“策略 → 收益”观测
EVENT_STRATEGY_MAP = {
    "external_insight_applied": "increase_weight",
    "evolution_target_selected": "select_evolution_target",
    "proposal_applied": "apply_proposal",
}


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_benefit(event: Dict[str, Any]) -> Optional[float]:
    """从事件中提取收益（优先元数据，其次消息里的“预期收益”。"""

    metadata = event.get("metadata") or {}
    for key in ("impact", "expected_gain", "benefit"):
        value = _as_float(metadata.get(key))
        if value is not None:
            return value

    before = _as_float(metadata.get("old_weight"))
    after = _as_float(metadata.get("new_weight"))
    if before is not None and after is not None:
        return after - before

    match = EXPECTED_GAIN_RE.search(event.get("message") or "")
    if match:
        return float(match.group(1)) / 100.0
    return None


@dataclass
class StrategyObservation:
    """一次可度量的策略应用记录。"""

    strategy_type: str
    target: str
    benefit: float
    source_event: str
    before: Optional[float] = None
    after: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "target": self.target,
            "benefit": round(self.benefit, 4),
            "source_event": self.source_event,
            "before": None if self.before is None else round(self.before, 4),
            "after": None if self.after is None else round(self.after, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class SimilarityObservation:
    """知识图谱相似度与实测成功率的配对观测。"""

    source_pattern: str
    target_pattern: str
    similarity: float
    success_rate: float
    domain: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pattern": self.source_pattern,
            "target_pattern": self.target_pattern,
            "similarity": round(self.similarity, 4),
            "success_rate": round(self.success_rate, 4),
            "domain": self.domain,
        }


@dataclass
class ResearchDataset:
    """统一格式的研究数据集，也是论文附录里的原始数据。"""

    generated_at: str = ""
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    similarity_pairs: List[Dict[str, Any]] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)
    pattern_success_rates: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    mental_model: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, bool] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        return {
            "snapshots": len(self.snapshots),
            "strategies": len(self.strategies),
            "phases": len(self.phases),
            "events": len(self.events),
            "similarity_pairs": len(self.similarity_pairs),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "counts": self.counts(),
            "sources": self.sources,
            "snapshots": self.snapshots,
            "strategies": self.strategies,
            "similarity_pairs": self.similarity_pairs,
            "phases": self.phases,
            "policy": self.policy,
            "pattern_success_rates": self.pattern_success_rates,
            "knowledge_graph": self.knowledge_graph,
            "mental_model": self.mental_model,
            "events": self.events,
        }


class _SnapshotProvider:
    """轻量适配器：让 MentalModelTrainer 复用已读出的快照，避免重复打开数据库。"""

    def __init__(self, snapshots: List[EvolutionSnapshot]):
        self._snapshots = snapshots

    def get_recent_snapshots(self, n: int = 100) -> List[EvolutionSnapshot]:
        return self._snapshots[-n:]


class ResearchDataCollector:
    """收集 Self-Researcher 所需的全部原始数据。"""

    def __init__(
        self,
        base_dir: str = ".",
        snapshot_db: Optional[str] = None,
        policy_path: Optional[str] = None,
        notification_log: Optional[str] = None,
        graph_path: Optional[str] = None,
        repo_path: Optional[str] = None,
    ):
        self._base_dir = base_dir
        self._snapshot_db = self._resolve(snapshot_db or DEFAULT_SNAPSHOT_DB)
        self._policy_path = self._resolve(policy_path or DEFAULT_POLICY_PATH)
        self._notification_log = self._resolve(notification_log or DEFAULT_NOTIFICATION_LOG)
        self._graph_path = self._resolve(graph_path or DEFAULT_GRAPH_PATH)
        self._repo_path = repo_path or base_dir

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path) or self._base_dir in ("", "."):
            return path
        return os.path.join(self._base_dir, path)

    def collect(self) -> ResearchDataset:
        snapshots = self._load_snapshots()
        events = self._load_events()
        graph = self._load_graph()

        return ResearchDataset(
            generated_at=datetime.now(timezone.utc).isoformat(),
            snapshots=[s.to_dict() for s in snapshots],
            strategies=[o.to_dict() for o in self._derive_strategies(events)],
            similarity_pairs=[p.to_dict() for p in self._derive_similarity_pairs(snapshots, graph)],
            phases=self._load_phases(),
            policy=self._load_policy(),
            pattern_success_rates=self._latest_pattern_success_rates(snapshots),
            events=events,
            knowledge_graph=graph,
            mental_model=self._train_mental_model(snapshots),
            sources={
                "snapshot_db": os.path.exists(self._snapshot_db),
                "policy": os.path.exists(self._policy_path),
                "notification_log": os.path.exists(self._notification_log),
                "knowledge_graph": os.path.exists(self._graph_path),
                "git_history": os.path.isdir(os.path.join(self._repo_path, ".git")),
            },
        )

    # ------------------------------------------------------------------ 数据源读取

    def _load_snapshots(self) -> List[EvolutionSnapshot]:
        if not os.path.exists(self._snapshot_db):
            return []
        try:
            return EvolutionTracker(self._snapshot_db).get_recent_snapshots(500)
        except Exception:
            return []

    def _load_events(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._notification_log):
            return []

        events: List[Dict[str, Any]] = []
        try:
            with open(self._notification_log, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return events

    def _load_policy(self) -> Dict[str, Any]:
        if not os.path.exists(self._policy_path):
            return {}
        try:
            return SelfImprovementPolicy(self._policy_path).load_policy().to_dict()
        except Exception:
            return {}

    def _load_graph(self) -> Dict[str, Any]:
        if not os.path.exists(self._graph_path):
            return {}
        try:
            amalgamator = KnowledgeAmalgamator(self._graph_path)
            amalgamator.load()
            return amalgamator.get_graph().to_dict()
        except Exception:
            return {}

    def _load_phases(self) -> List[Dict[str, Any]]:
        try:
            return [stage.to_dict() for stage in SelfRepositoryMiner(self._repo_path).mine_self()]
        except Exception:
            return []

    def _train_mental_model(self, snapshots: List[EvolutionSnapshot]) -> Dict[str, Any]:
        if len(snapshots) < 2:
            return {}
        try:
            trainer = MentalModelTrainer(_SnapshotProvider(snapshots))
            return trainer.train().to_dict()
        except Exception:
            return {}

    # ------------------------------------------------------------------ 派生观测

    def _derive_strategies(self, events: List[Dict[str, Any]]) -> List[StrategyObservation]:
        observations: List[StrategyObservation] = []

        for event in events:
            event_type = event.get("event_type")
            metadata = event.get("metadata") or {}

            strategy_type = EVENT_STRATEGY_MAP.get(event_type or "")
            if event_type == "plan_executed":
                strategy_type = metadata.get("selected_strategy") or "plan_executed"
            if not strategy_type:
                continue

            benefit = extract_benefit(event)
            if benefit is None:
                continue

            observations.append(
                StrategyObservation(
                    strategy_type=strategy_type,
                    target=str(metadata.get("target") or metadata.get("pattern") or "unknown"),
                    benefit=benefit,
                    source_event=str(event_type),
                    before=_as_float(metadata.get("old_weight")),
                    after=_as_float(metadata.get("new_weight")),
                    timestamp=event.get("timestamp"),
                )
            )

        return observations

    def _derive_similarity_pairs(
        self, snapshots: List[EvolutionSnapshot], graph: Dict[str, Any]
    ) -> List[SimilarityObservation]:
        pattern_rates = self._latest_pattern_success_rates(snapshots)
        if not pattern_rates:
            return []

        node_patterns = {node.get("id"): node.get("pattern_type") for node in graph.get("nodes", [])}
        pairs: List[SimilarityObservation] = []

        for edge in graph.get("edges", []):
            similarity = _as_float(edge.get("similarity"))
            if similarity is None:
                continue

            source_pattern = node_patterns.get(edge.get("source_id"))
            target_pattern = node_patterns.get(edge.get("target_id"))

            success_rate = None
            for name in (target_pattern, source_pattern):
                if name and name in pattern_rates:
                    success_rate = pattern_rates[name]
                    break
            if success_rate is None:
                continue

            pairs.append(
                SimilarityObservation(
                    source_pattern=source_pattern or "unknown",
                    target_pattern=target_pattern or "unknown",
                    similarity=similarity,
                    success_rate=success_rate,
                    domain="+".join(edge.get("domains") or []) or "unknown",
                )
            )

        return pairs

    def _latest_pattern_success_rates(self, snapshots: List[EvolutionSnapshot]) -> Dict[str, float]:
        for snapshot in reversed(snapshots):
            rates = (snapshot.metadata or {}).get("pattern_success_rates")
            if rates:
                return {str(k): float(v) for k, v in rates.items()}
        return {}
