from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import shutil

from .evolution_tracker import EvolutionTracker
from .knowledge_amalgamator import KnowledgeAmalgamator
from .self_improvement_policy import SelfImprovementPolicy
from .trend_detector import TrendDetector


def _format_bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    filled = max(1, int((value / max_val) * width))
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}]"


def _trend_arrow(values: List[float]) -> str:
    if len(values) < 2:
        return "->"
    last_two = values[-2:]
    diff = last_two[1] - last_two[0]
    if diff > 0.01:
        return "^"
    elif diff < -0.01:
        return "v"
    return "->"


def _status_str(sr: float) -> str:
    if sr >= 0.70:
        return "[OK]"
    elif sr >= 0.50:
        return "[!]"
    return "[ERR]"


def render_dashboard(tracker: Optional[EvolutionTracker] = None,
                     amalgamator: Optional[KnowledgeAmalgamator] = None,
                     policy_loader: Optional[SelfImprovementPolicy] = None) -> str:
    snapshots = []
    if tracker:
        snapshots = tracker.get_recent_snapshots(10)

    total_patterns = 0
    total_relations = 0
    kg_graph = None
    if amalgamator:
        total_patterns = amalgamator.get_pattern_count()
        total_relations = amalgamator.get_relationship_count()
        kg_graph = amalgamator.get_graph()

    current_policy = None
    if policy_loader:
        current_policy = policy_loader.load_policy()

    terminal_width = min(shutil.get_terminal_size().columns, 100)
    sep = "-" * terminal_width

    lines = []
    lines.append("")
    lines.append(f"{'=' * terminal_width}")
    lines.append(f"  AutoTestGen 3.0 演化仪表盘")
    lines.append(f"{'=' * terminal_width}")
    lines.append("")

    if snapshots:
        latest = snapshots[-1]
        sr_values = [s.success_rate for s in snapshots]
        cd_values = [s.cross_domain_success for s in snapshots]
        sr_trend = _trend_arrow(sr_values)
        cd_trend = _trend_arrow(cd_values)

        time_ago = datetime.now(timezone.utc) - latest.timestamp
        hours_ago = int(time_ago.total_seconds() / 3600)

        status_label = _status_str(latest.success_rate)
        rising = sum(1 for i in range(1, len(sr_values)) if sr_values[i] >= sr_values[i-1])
        falling = len(sr_values) - 1 - rising
        trend_text = f"连续 {rising} 次上升" if rising > falling else f"连续 {falling} 次下降"

        lines.append(f"  [*] 当前状态: {status_label}")
        lines.append(f"  |- 总体成功率:    {latest.success_rate:.2f} {sr_trend}  {_format_bar(latest.success_rate)}")
        lines.append(f"  |- 跨域迁移率:    {latest.cross_domain_success:.2f} {cd_trend}  {_format_bar(latest.cross_domain_success)}")
        lines.append(f"  |- 模式覆盖度:    {latest.pattern_coverage:.2f}    {_format_bar(latest.pattern_coverage)}")
        lines.append(f"  |- 知识图谱:      {total_patterns} 种模式, {total_relations} 条跨域链接")
        lines.append(f"  |- 最近快照:      #{latest.snapshot_index} ({hours_ago} 小时前)")
        lines.append(f"  +- 趋势:          {trend_text} ({sr_values[-1] - sr_values[0]:+.2f})")
    else:
        lines.append("  暂无快照数据")

    lines.append("")
    lines.append(sep)

    if snapshots and len(snapshots) >= 3:
        sr_values = [s.success_rate for s in snapshots]
        lines.append("  [成功率趋势]")
        if len(sr_values) >= 2:
            max_val = max(sr_values) if max(sr_values) > 0 else 1
            for i in range(0, len(sr_values), max(1, len(sr_values) // 5)):
                v = sr_values[i]
                bar_len = max(1, int((v / max_val) * 25))
                bar = "#" * bar_len
                label = f"#{snapshots[i].snapshot_index}"
                lines.append(f"    {label:>6s} |{bar} {v:.2f}")
        lines.append("")

    lines.append(f"  [最近事件]")
    events = _build_recent_events(snapshots)
    for event in events[:5]:
        lines.append(f"    {event}")
    lines.append("")

    lines.append(sep)
    lines.append("  [薄弱环节]")
    if kg_graph:
        weak_nodes = sorted(kg_graph.nodes, key=lambda n: n.confidence)[:3]
        for node in weak_nodes:
            lines.append(f"    - '{node.pattern_type.value}' 置信度 {node.confidence:.0%}, 出现 {node.occurrences} 次")
        if weak_nodes:
            w = weak_nodes[0]
            lines.append(f"    建议: 增加 {w.pattern_type.value} 的采样权重 (当前 {w.confidence:.1f} -> {min(1.0, w.confidence + 0.2):.1f})")
    lines.append("")

    lines.append(sep)
    lines.append("  [下一步计划]")
    if current_policy:
        high_priority = [(pt, w) for pt, w in sorted(current_policy.pattern_weights.items(), key=lambda x: x[1], reverse=True) if w > 0.2]
        if high_priority:
            lines.append(f"    - 优先处理 '{high_priority[0][0]}' 相关案例 (权重 {high_priority[0][1]:.2f})")
        lines.append(f"    - 变异率: {current_policy.mutation_rate}, 探索因子: {current_policy.exploration_factor}")
        if snapshots:
            lines.append(f"    - 计划在 #{snapshots[-1].snapshot_index + 1} 快照后重新评估")
    lines.append("")
    lines.append(f"{'=' * terminal_width}")
    lines.append("")

    return "\n".join(lines)


def _build_recent_events(snapshots: List) -> List[str]:
    events = []
    for i, snap in enumerate(snapshots):
        date_str = snap.timestamp.strftime("%Y-%m-%d %H:%M")
        meta = snap.metadata or {}

        if meta.get("corrected"):
            events.append(f"{date_str}: [OK] 修正 'multiple' 自动修复完成")
            continue

        pattern_rates = meta.get("pattern_success_rates", {})
        for ptype, rate in pattern_rates.items():
            if rate > 0.85:
                events.append(f"{date_str}: [*] 发现新模式 '{ptype}'")
                break

        if i > 0:
            delta = snap.success_rate - snapshots[i - 1].success_rate
            if delta > 0.03:
                events.append(f"{date_str}: [^] 实验 #{snap.snapshot_index} 通过率 {snap.success_rate:.0%}")
            elif delta < -0.03:
                events.append(f"{date_str}: [!] 实验 #{snap.snapshot_index} 通过率降至 {snap.success_rate:.0%}")

        if i == len(snapshots) - 1 and snap.success_rate >= 0.80:
            events.append(f"{date_str}: [*] 系统达成最高成功率 {snap.success_rate:.0%}")

    return events[::-1][:8]


def start_interactive(tracker: Optional[EvolutionTracker] = None,
                      amalgamator: Optional[KnowledgeAmalgamator] = None,
                      policy_loader: Optional[SelfImprovementPolicy] = None) -> None:
    import time
    try:
        while True:
            output = render_dashboard(tracker, amalgamator, policy_loader)
            print(output)

            exit_cmds = {"q", "quit", "exit"}
            cmd = input("  输入 q 退出, 回车刷新: ").strip().lower()
            if cmd in exit_cmds:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n  仪表盘已关闭。")