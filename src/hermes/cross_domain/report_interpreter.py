from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re


@dataclass
class Signal:
    signal_type: str
    pattern_type: Optional[str]
    direction: str  # up, down, stable
    magnitude: float
    metric: str
    description: str
    confidence: float
    source: str = "report"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "magnitude": round(self.magnitude, 2),
            "metric": self.metric,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "source": self.source,
        }


@dataclass
class SignalReport:
    signals: List[Signal] = field(default_factory=list)
    up_signals: int = 0
    down_signals: int = 0
    anomalies: int = 0
    total_metrics_analyzed: int = 0
    report_snapshot_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self.signals),
            "up_signals": self.up_signals,
            "down_signals": self.down_signals,
            "anomalies": self.anomalies,
            "total_metrics_analyzed": self.total_metrics_analyzed,
            "report_snapshot_index": self.report_snapshot_index,
            "signals": [s.to_dict() for s in self.signals],
        }


SIGNAL_KEYWORDS = {
    "success_rate": ["成功率", "success_rate", "通过率"],
    "cross_domain_success": ["跨域迁移", "cross_domain", "迁移率"],
    "pattern_coverage": ["覆盖度", "coverage", "覆盖"],
    "type_mismatch": ["type_mismatch", "类型不匹配"],
    "boundary_check_missing": ["boundary_check", "边界检查"],
    "resource_limit_missing": ["resource_limit", "资源限制"],
    "concurrency_race_condition": ["concurrency_race", "竞态"],
    "infinite_recursion": ["infinite_recursion", "递归"],
    "memory_leak": ["memory_leak", "内存泄漏"],
}


class ReportInterpreter:
    def __init__(self):
        self._last_report_path: Optional[str] = None

    def interpret_path(self, report_path: str) -> SignalReport:
        self._last_report_path = report_path
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._parse_content(content)
        except FileNotFoundError:
            return SignalReport()

    def interpret_content(self, content: str) -> SignalReport:
        return self._parse_content(content)

    def _parse_content(self, content: str) -> SignalReport:
        sr = SignalReport()
        lines = content.split("\n")

        for line in lines:
            sig = self._try_parse_metric_line(line)
            if sig:
                sr.signals.append(sig)

            sig = self._try_parse_weakness_line(line)
            if sig:
                sr.signals.append(sig)

            sig = self._try_parse_trend_line(line)
            if sig:
                sr.signals.append(sig)

        sr.up_signals = sum(1 for s in sr.signals if s.direction == "up")
        sr.down_signals = sum(1 for s in sr.signals if s.direction == "down")
        sr.anomalies = sum(1 for s in sr.signals if s.magnitude > 0.1 and s.direction == "down")
        sr.total_metrics_analyzed = len(sr.signals)

        snapshot_match = re.search(r'#(\d+)', content)
        if snapshot_match:
            sr.report_snapshot_index = int(snapshot_match.group(1))

        return sr

    def _try_parse_metric_line(self, line: str) -> Optional[Signal]:
        for metric_name, keywords in SIGNAL_KEYWORDS.items():
            if any(kw in line for kw in keywords):
                direction = "up" if "↑" in line or "提升" in line or "+" in line else "down" if "↓" in line or "下降" in line else "stable"

                value_match = re.search(r'(\d+\.?\d*)%', line)
                if value_match:
                    magnitude = float(value_match.group(1)) / 100.0
                else:
                    magnitude = 0.0

                return Signal(
                    signal_type="metric",
                    pattern_type=metric_name if metric_name in SIGNAL_KEYWORDS else None,
                    direction=direction,
                    magnitude=magnitude,
                    metric=metric_name,
                    description=f"{metric_name}: {line.strip()}",
                    confidence=0.7 if direction != "stable" else 0.4,
                )
        return None

    def _try_parse_weakness_line(self, line: str) -> Optional[Signal]:
        if "薄弱" in line or "建议" in line or "置信度" in line:
            ptype_match = re.search(r"'(\w+)'", line)
            conf_match = re.search(r'(\d+)%', line)

            ptype = ptype_match.group(1) if ptype_match else "unknown"
            conf = float(conf_match.group(1)) / 100.0 if conf_match else 0.5

            return Signal(
                signal_type="weakness",
                pattern_type=ptype,
                direction="down",
                magnitude=1.0 - conf,
                metric=ptype,
                description=line.strip(),
                confidence=0.6,
            )
        return None

    def _try_parse_trend_line(self, line: str) -> Optional[Signal]:
        trend_match = re.search(r'趋势.*?(上升|下降)', line)
        if trend_match:
            direction_str = trend_match.group(1)
            direction = "up" if "上升" in direction_str else "down"

            value_match = re.search(r'([+-]\d+\.\d+)', line)
            magnitude = float(value_match.group(1)) if value_match else 0.02

            return Signal(
                signal_type="trend",
                pattern_type=None,
                direction=direction,
                magnitude=magnitude,
                metric="overall_trend",
                description=line.strip(),
                confidence=0.5,
            )
        return None