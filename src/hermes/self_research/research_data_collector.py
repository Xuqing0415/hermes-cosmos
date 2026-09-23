"""Self-Researcher 数据收集层。

论文需要的数据分散在多个既有子系统中，本模块把它们收敛成统一的数据集：

* 演化快照：`EvolutionTracker` 的 SQLite 表
* 策略事件：`loop_output/notifications.log` 的 JSON Lines 日志
* 改进策略：`SelfImprovementPolicy` 的 JSON 策略文件
* 知识图谱：`KnowledgeAmalgamator` 的 JSON 图谱
* 演化阶段：`GitPhaseDetector` 对自身 Git 历史的划分（git 不可用时不划分，也不伪造）

所有数据源都是可选的：文件缺失或损坏时退化为空集合而不是抛错，
这样在没有任何历史产物的环境下也能跑完整条论文生成流水线。

采集的同时会给每个字段盖上来源标签（见 `data_provenance`），
让下游能区分“真实数据”和“降级/合成数据”。
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from hermes.cross_domain.evolution_tracker import EvolutionSnapshot, EvolutionTracker
from hermes.cross_domain.git_phase_detector import GitPhaseDetector, PhaseDetectionResult
from hermes.cross_domain.knowledge_amalgamator import KnowledgeAmalgamator
from hermes.cross_domain.mental_model_trainer import MentalModelTrainer
from hermes.cross_domain.self_improvement_policy import SelfImprovementPolicy
from hermes.self_research.data_provenance import REAL, UNVERIFIED

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
    phase_detection: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    provenance_notes: Dict[str, str] = field(default_factory=dict)
    provenance_counts: Dict[str, int] = field(default_factory=dict)

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
            "phase_detection": self.phase_detection,
            "provenance": self.provenance,
            "provenance_notes": self.provenance_notes,
            "provenance_counts": self.provenance_counts,
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
        phase_detector: Optional[GitPhaseDetector] = None,
        provenance: Optional[Dict[str, str]] = None,
    ):
        self._base_dir = base_dir
        self._snapshot_db = self._resolve(snapshot_db or DEFAULT_SNAPSHOT_DB)
        self._policy_path = self._resolve(policy_path or DEFAULT_POLICY_PATH)
        self._notification_log = self._resolve(notification_log or DEFAULT_NOTIFICATION_LOG)
        self._graph_path = self._resolve(graph_path or DEFAULT_GRAPH_PATH)
        self._repo_path = repo_path or base_dir
        self._phase_detector = phase_detector or GitPhaseDetector(self._repo_path)
        self._forced_provenance = dict(provenance or {})

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path) or self._base_dir in ("", "."):
            return path
        return os.path.join(self._base_dir, path)

    def collect(self) -> ResearchDataset:
        snapshots = self._load_snapshots()
        events = self._load_events()
        graph = self._load_graph()
        policy = self._load_policy()
        phase_result = self._detect_phases()
        mental_model = self._train_mental_model(snapshots)
        strategies = self._derive_strategies(events)
        pattern_rates = self._latest_pattern_success_rates(snapshots)
        similarity_pairs = self._derive_similarity_pairs(snapshots, graph)

        sources = {
            "snapshot_db": os.path.exists(self._snapshot_db),
            "policy": os.path.exists(self._policy_path),
            "notification_log": os.path.exists(self._notification_log),
            "knowledge_graph": os.path.exists(self._graph_path),
            "git_history": not phase_result.git_unavailable,
        }
        counts = {
            "snapshots": len(snapshots),
            "events": len(events),
            "strategies": len(strategies),
            "similarity_pairs": len(similarity_pairs),
            "policy": len(policy.get("pattern_weights") or {}),
            "knowledge_graph": len(graph.get("nodes") or []),
            "pattern_success_rates": len(pattern_rates),
            "mental_model": int(mental_model.get("training_samples", 0) or 0),
            "phases": len(phase_result.phases),
        }
        provenance, provenance_notes = self._stamp_provenance(
            sources=sources,
            phase_result=phase_result,
            counts=counts,
            mental_model=mental_model,
        )

        return ResearchDataset(
            generated_at=datetime.now(timezone.utc).isoformat(),
            snapshots=[s.to_dict() for s in snapshots],
            strategies=[o.to_dict() for o in strategies],
            similarity_pairs=[p.to_dict() for p in similarity_pairs],
            phases=[phase.to_dict() for phase in phase_result.phases],
            policy=policy,
            pattern_success_rates=pattern_rates,
            events=events,
            knowledge_graph=graph,
            mental_model=mental_model,
            sources=sources,
            phase_detection=phase_result.to_dict(),
            provenance=provenance,
            provenance_notes=provenance_notes,
            provenance_counts=counts,
        )

    # ------------------------------------------------------------------ 来源标注

    def _stamp_provenance(
        self,
        sources: Dict[str, bool],
        phase_result: PhaseDetectionResult,
        counts: Dict[str, int],
        mental_model: Dict[str, Any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """给每个字段盖上来源标签。

        原则：`REAL` 只能用在“数据确实来自可外部核对的来源”上。数据源不可用时字段是
        空的，标签必须是 `UNVERIFIED`（来源不可核对）并写明“无数据、未做任何推断”：
        给空数据盖 `REAL`，等于宣称一份并不存在的数据来自真实来源。
        """

        provenance: Dict[str, str] = {}
        notes: Dict[str, str] = {}

        def stamp(field_name: str, source: str, real_detail: str, missing_detail: str) -> None:
            if sources.get(source):
                provenance[field_name] = REAL
                notes[field_name] = f"{counts.get(field_name, 0)} 条来自{real_detail}"
            else:
                provenance[field_name] = UNVERIFIED
                notes[field_name] = f"无数据：{missing_detail}，未做任何推断"

        stamp("snapshots", "snapshot_db", f" SQLite 快照库（{self._snapshot_db}）", "快照库不可用")
        stamp("events", "notification_log", f" 事件日志（{self._notification_log}）", "事件日志不可用")
        stamp("strategies", "notification_log", " 事件日志中带收益字段的事件", "事件日志不可用")
        stamp("policy", "policy", f" 改进策略文件（{self._policy_path}）", "策略文件不可用")
        stamp("knowledge_graph", "knowledge_graph", f" 知识图谱文件（{self._graph_path}）", "知识图谱不可用")
        stamp("pattern_success_rates", "snapshot_db", " 快照元数据中的模式成功率", "快照库不可用")
        stamp("mental_model", "snapshot_db", " 快照上的心智模型训练", "快照库不可用")
        notes["mental_model"] = self._mental_model_note(sources, mental_model)

        if sources.get("knowledge_graph") and sources.get("snapshot_db"):
            provenance["similarity_pairs"] = REAL
            notes["similarity_pairs"] = f"{counts.get('similarity_pairs', 0)} 组来自知识图谱边与快照模式成功率的配对"
        else:
            provenance["similarity_pairs"] = UNVERIFIED
            notes["similarity_pairs"] = "无数据：知识图谱或快照库不可用，未做任何推断"

        if sources.get("git_history"):
            provenance["phases"] = REAL
            notes["phases"] = (
                f"{len(phase_result.phases)} 个阶段由真实 git 历史推导"
                f"（{phase_result.commit_count} 次提交，方法 {phase_result.method}）"
            )
        else:
            provenance["phases"] = UNVERIFIED
            notes["phases"] = f"无数据：{phase_result.reason}，本次不划分演化阶段"

        for field_name, label in self._forced_provenance.items():
            provenance[field_name] = label
            notes[field_name] = f"调用方声明为 {label}"

        return provenance, notes

    @staticmethod
    def _mental_model_note(sources: Dict[str, bool], mental_model: Dict[str, Any]) -> str:
        """心智模型的来源备注：样本不足时必须说明“未做留出评测”。"""

        if not sources.get("snapshot_db"):
            return "无数据：快照库不可用，未训练心智模型，未做任何推断"

        samples = int(mental_model.get("training_samples", 0) or 0)
        if not mental_model:
            return "无数据：快照不足以训练心智模型，未做任何推断"
        if mental_model.get("estimated"):
            return f"{samples} 个真实快照：样本不足，未做留出评测，不报告准确率"
        if not mental_model.get("reliable", False):
            return f"{samples} 个真实快照：留出评测样本不足，准确率不具统计意义，论文不引用"
        return f"{samples} 个真实快照上的 MentalModelTrainer 留出评测"

    def _detect_phases(self) -> PhaseDetectionResult:
        try:
            return self._phase_detector.detect()
        except Exception as error:
            return PhaseDetectionResult(
                git_unavailable=True,
                reason=f"阶段检测失败：{error}",
                method="unavailable",
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
