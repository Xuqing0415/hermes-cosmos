from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os

from .evolution_tracker import EvolutionTracker
from .knowledge_amalgamator import KnowledgeAmalgamator
from .self_improvement_policy import SelfImprovementPolicy, EvolutionPolicy
from .trend_detector import TrendDetector, TrendReport


REPORT_HEADER = """# AutoTestGen 3.0 演化报告

> 生成时间: {timestamp}
> 系统运行: {total_snapshots} 次快照, {total_patterns} 种模式, {total_relations} 条跨域链接

---

"""

SECTION_CAPABILITY = """
## 1. 能力概览

| 指标 | 当前值 | 最近趋势 | 状态 |
|------|--------|----------|------|
| 总体成功率 | {success_rate:.0%} | {success_trend} | {success_status} |
| 跨领域迁移率 | {cross_domain:.0%} | {cross_trend} | {cross_status} |
| 模式覆盖度 | {coverage:.0%} | {coverage_trend} | {coverage_status} |

"""

SECTION_MILESTONES = """
## 2. 关键里程碑

{events}

"""

SECTION_WEAKNESS = """
## 3. 薄弱环节分析

| 模式类型 | 成功率 | 建议改进 |
|----------|--------|----------|
{weakness_rows}

"""

SECTION_NEXT_PLAN = """
## 4. 下一步演化计划

{plan}

### 当前策略参数

| 参数 | 当前值 |
|------|--------|
| 变异率 | {mutation_rate} |
| 探索因子 | {exploration_factor} |
| 置信度阈值 | {confidence_threshold} |
| 策略更新次数 | {update_count} |

### 模式权重

| 模式 | 权重 |
|------|------|
{weight_rows}

"""

SECTION_ASCII_CHART = """
## 5. 成功率趋势

```
{chart}
```

"""

EVENT_TEMPLATES = {
    "pattern_discovery": "- {date}: **发现新模式** '{pattern}' (来自 {domain} 领域)",
    "correction": "- {date}: **修正** '{pattern}' {action}",
    "experiment": "- {date}: 实验 #{snapshot} 通过率 {rate:.0%}",
    "degradation": "- {date}: **退化告警** {metric} 连续 {count} 次下降",
    "milestone": "- {date}: **里程碑** 系统达成 {metric} = {value:.0%}",
    "transfer_breakthrough": "- {date}: **突破** 实现了 {source} -> {target} 的跨领域迁移",
}


def _format_trend(values: List[float]) -> str:
    if len(values) < 2:
        return "——"
    recent = values[-3:] if len(values) >= 3 else values
    if len(recent) < 2:
        return "稳定"

    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    avg_delta = sum(deltas) / len(deltas)

    if avg_delta > 0.02:
        return f"↑ {avg_delta:+.0%}"
    elif avg_delta < -0.02:
        return f"↓ {avg_delta:+.0%}"
    else:
        return "→ 稳定"


def _status_icon(trend_str: str) -> str:
    if "↑" in trend_str:
        return "[OK] 提升"
    elif "↓" in trend_str:
        return "[!] 下降"
    return "[OK] 正常"


def _ascii_bar_chart(values: List[float], width: int = 30, label: str = "值") -> str:
    if not values:
        return "(无数据)"

    max_val = max(values) if max(values) > 0 else 1
    lines = []
    for i, v in enumerate(values):
        bar_len = max(1, int((v / max_val) * width))
        bar = "█" * bar_len
        lines.append(f"{label} #{i + 1:2d} │{bar} {v:.2f}")
    return "\n".join(lines)


class ReportGenerator:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 amalgamator: Optional[KnowledgeAmalgamator] = None,
                 policy: Optional[SelfImprovementPolicy] = None):
        self._tracker = tracker
        self._amalgamator = amalgamator
        self._policy = policy

    def generate_report(self, output_path: str = "evolution_report.md") -> str:
        snapshots = []
        if self._tracker:
            snapshots = self._tracker.get_recent_snapshots(20)

        total_snapshots = len(snapshots)
        total_patterns = 0
        total_relations = 0
        if self._amalgamator:
            total_patterns = self._amalgamator.get_pattern_count()
            total_relations = self._amalgamator.get_relationship_count()

        kg_graph = self._amalgamator.get_graph() if self._amalgamator else None

        current_policy = EvolutionPolicy()
        if self._policy:
            current_policy = self._policy.load_policy()

        current_sr = snapshots[-1].success_rate if snapshots else 0.0
        current_cd = snapshots[-1].cross_domain_success if snapshots else 0.0
        current_pc = snapshots[-1].pattern_coverage if snapshots else 0.0

        sr_values = [s.success_rate for s in snapshots]
        cd_values = [s.cross_domain_success for s in snapshots]
        pc_values = [s.pattern_coverage for s in snapshots]

        success_trend = _format_trend(sr_values)
        cross_trend = _format_trend(cd_values)
        coverage_trend = _format_trend(pc_values)

        content = REPORT_HEADER.format(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_snapshots=total_snapshots,
            total_patterns=total_patterns,
            total_relations=total_relations,
        )

        content += SECTION_CAPABILITY.format(
            success_rate=current_sr,
            cross_domain=current_cd,
            coverage=current_pc,
            success_trend=success_trend,
            cross_trend=cross_trend,
            coverage_trend=coverage_trend,
            success_status=_status_icon(success_trend),
            cross_status=_status_icon(cross_trend),
            coverage_status=_status_icon(coverage_trend),
        )

        events = self._build_events(snapshots)
        content += SECTION_MILESTONES.format(events=events or "(暂无重大事件记录)")

        weakness_rows = self._build_weakness_rows(kg_graph)
        content += SECTION_WEAKNESS.format(weakness_rows=weakness_rows)

        weight_rows = "\n".join(
            f"| {pt} | {w:.2f} |"
            for pt, w in sorted(current_policy.pattern_weights.items(), key=lambda x: x[1], reverse=True)
        )

        next_action = ""
        if self._policy:
            from .gap_analyzer import GapReport
            mock_gap = GapReport(overall_success_rate=current_sr * 100)
            next_action = self._policy.recommend_next_action(current_policy, mock_gap)

        content += SECTION_NEXT_PLAN.format(
            plan=next_action or "继续当前演化策略",
            mutation_rate=current_policy.mutation_rate,
            exploration_factor=current_policy.exploration_factor,
            confidence_threshold=current_policy.min_confidence_threshold,
            update_count=current_policy.update_count,
            weight_rows=weight_rows,
        )

        chart = _ascii_bar_chart(sr_values, width=25, label="快照")
        content += SECTION_ASCII_CHART.format(chart=chart)

        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path

    def _build_events(self, snapshots: List) -> str:
        events = []
        for i, snap in enumerate(snapshots):
            date_str = snap.timestamp.strftime("%Y-%m-%d %H:%M")
            meta = snap.metadata or {}

            if meta.get("corrected"):
                events.append(EVENT_TEMPLATES["correction"].format(
                    date=date_str, pattern="multiple", action="自动修复完成"
                ))

            if i > 0 and snap.success_rate > snapshots[i - 1].success_rate + 0.05:
                events.append(EVENT_TEMPLATES["experiment"].format(
                    date=date_str, snapshot=snap.snapshot_index, rate=snap.success_rate
                ))

            pattern_rates = meta.get("pattern_success_rates", {})
            for ptype, rate in pattern_rates.items():
                if rate > 0.85:
                    events.append(EVENT_TEMPLATES["pattern_discovery"].format(
                        date=date_str, pattern=ptype, domain=",".join(snap.domains if hasattr(snap, 'domains') else ["unknown"])
                    ))
                    break

        return "\n".join(events[:10]) if events else "(暂无重大事件记录)"

    def _build_weakness_rows(self, graph) -> str:
        if not graph:
            return "| - | - | - |\n"
        rows = []
        for node in graph.nodes:
            if node.confidence < 0.6:
                rows.append(
                    f"| {node.pattern_type.value} | {node.confidence:.0%} 置信度 | "
                    f"增加采样权重 (当前 {node.occurrences} 次出现) |"
                )
        if not rows:
            for node in graph.nodes[:3]:
                rows.append(
                    f"| {node.pattern_type.value} | {node.confidence:.0%} 置信度 | 继续监测 |"
                )
        return "\n".join(rows) if rows else "| - | - | - |\n"