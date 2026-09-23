"""数据来源标注。

论文里每一个数字都必须能回答一个问题：它从哪来？本模块给数据集的每个字段
打上来源标签：

* `REAL`        —— 来自可外部核对的持久化数据源（git 对象、SQLite 行、日志行、JSON 文件）
* `FALLBACK`    —— 主数据源不可用时由替代/降级路径产生
* `SYNTHETIC`   —— 演示或人工合成数据
* `UNVERIFIED`  —— 采集层没有标注来源；按不可信处理（评级时等同 FALLBACK）

标签由采集层在收集数据时显式盖上（`ResearchDataset.provenance`），
没有标签的数据不会被默认为真实数据。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

REAL = "REAL"
FALLBACK = "FALLBACK"
SYNTHETIC = "SYNTHETIC"
UNVERIFIED = "UNVERIFIED"

ALL_PROVENANCE = (REAL, FALLBACK, SYNTHETIC, UNVERIFIED)

PROVENANCE_LABELS = {
    REAL: "真实数据",
    FALLBACK: "降级数据",
    SYNTHETIC: "合成数据",
    UNVERIFIED: "来源未标注",
}

_PROVENANCE_RANK = {REAL: 0, FALLBACK: 1, UNVERIFIED: 2, SYNTHETIC: 3}

# 数据集字段 -> 它依赖的主数据源
FIELD_SOURCES = {
    "snapshots": "snapshot_db",
    "events": "notification_log",
    "strategies": "notification_log",
    "pattern_success_rates": "snapshot_db",
    "similarity_pairs": "knowledge_graph",
    "policy": "policy",
    "knowledge_graph": "knowledge_graph",
    "mental_model": "snapshot_db",
    "phases": "git_history",
}

SOURCE_LABELS = {
    "snapshot_db": "SQLite 快照库",
    "notification_log": "事件日志",
    "policy": "改进策略文件",
    "knowledge_graph": "知识图谱",
    "git_history": "git 历史",
}


def worst(provenances: Iterable[str]) -> str:
    """取一组来源标签里最不可信的那个（顺序无关，结果确定）。"""

    items = [item for item in provenances if item in _PROVENANCE_RANK]
    if not items:
        return UNVERIFIED
    return sorted(items, key=lambda item: (_PROVENANCE_RANK[item], item))[-1]


@dataclass
class ProvenanceEntry:
    field: str
    provenance: str
    detail: str
    count: int = 0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "provenance": self.provenance,
            "label": PROVENANCE_LABELS.get(self.provenance, self.provenance),
            "detail": self.detail,
            "count": self.count,
            "source": self.source,
        }


@dataclass
class ProvenanceReport:
    entries: Dict[str, ProvenanceEntry] = field(default_factory=dict)
    missing_sources: List[str] = field(default_factory=list)
    git_unavailable: bool = False
    notes: List[str] = field(default_factory=list)

    def provenance_of(self, field_name: str) -> str:
        entry = self.entries.get(field_name)
        return entry.provenance if entry else UNVERIFIED

    def worst_for(self, field_names: Iterable[str]) -> str:
        names = list(field_names)
        if not names:
            return UNVERIFIED
        return worst(self.provenance_of(name) for name in names)

    def count_by_provenance(self) -> Dict[str, int]:
        counts = dict.fromkeys(ALL_PROVENANCE, 0)
        for entry in self.entries.values():
            counts[entry.provenance] = counts.get(entry.provenance, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "missing_sources": self.missing_sources,
            "git_unavailable": self.git_unavailable,
            "notes": self.notes,
            "counts_by_provenance": self.count_by_provenance(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProvenanceReport":
        """从 `to_dict()` 的结果还原，用于在只拿到落盘产物时重算覆盖率。"""

        entries: Dict[str, ProvenanceEntry] = {}
        for item in (payload or {}).get("entries") or []:
            field_name = str(item.get("field") or "")
            if not field_name:
                continue
            entries[field_name] = ProvenanceEntry(
                field=field_name,
                provenance=str(item.get("provenance") or UNVERIFIED),
                detail=str(item.get("detail") or ""),
                count=int(item.get("count") or 0),
                source=str(item.get("source") or ""),
            )
        return cls(
            entries=entries,
            missing_sources=[str(item) for item in (payload or {}).get("missing_sources") or []],
            git_unavailable=bool((payload or {}).get("git_unavailable")),
            notes=[str(item) for item in (payload or {}).get("notes") or []],
        )


class DataProvenanceTagger:
    """把采集层盖好的来源标签整理成报告；未标注的字段不会被当成真实数据。"""

    def __init__(self, field_sources: Optional[Dict[str, str]] = None):
        self._field_sources = dict(field_sources or FIELD_SOURCES)

    def tag(self, dataset: Any) -> ProvenanceReport:
        stamped: Dict[str, str] = dict(getattr(dataset, "provenance", None) or {})
        notes: Dict[str, str] = dict(getattr(dataset, "provenance_notes", None) or {})
        sources: Dict[str, bool] = dict(getattr(dataset, "sources", None) or {})

        entries: Dict[str, ProvenanceEntry] = {}
        for field_name, source in self._field_sources.items():
            count = self._count_field(dataset, field_name)
            label = stamped.get(field_name)

            if label is None:
                if count == 0:
                    label = REAL
                    default_detail = "无数据：该数据源不可用，未做任何推断"
                else:
                    label = UNVERIFIED
                    default_detail = "采集层未标注来源"
            else:
                default_detail = "采集层标注"

            detail = notes.get(field_name) or default_detail
            entries[field_name] = ProvenanceEntry(
                field=field_name,
                provenance=label,
                detail=detail,
                count=count,
                source=source,
            )

        # 以“已知的全部数据源”为准：采集层没声明的源同样算缺失，
        # 否则一份什么都没声明的数据集会被算成覆盖率 100%
        missing = sorted(name for name in SOURCE_LABELS if not sources.get(name))
        report = ProvenanceReport(
            entries=entries,
            missing_sources=missing,
            git_unavailable=not bool(sources.get("git_history", False)),
        )

        for name in missing:
            report.notes.append(f"数据源缺失：{SOURCE_LABELS.get(name, name)}")
        for field_name, entry in entries.items():
            if entry.provenance in (FALLBACK, SYNTHETIC, UNVERIFIED) and entry.count > 0:
                report.notes.append(f"{field_name} 的来源是 {PROVENANCE_LABELS[entry.provenance]}：{entry.detail}")

        return report

    @staticmethod
    def _count_field(dataset: Any, field_name: str) -> int:
        stamped_counts: Dict[str, int] = dict(getattr(dataset, "provenance_counts", None) or {})
        if field_name in stamped_counts:
            return int(stamped_counts[field_name])
        value = getattr(dataset, field_name, None)
        if value is None:
            return 0
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        return 1
