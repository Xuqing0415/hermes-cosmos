"""Self-Researcher 发现提取层。

把统计结果按固定规则翻译成“可验证的规律”。规则是确定性的、可复现的，
每条发现都必须带上证据、样本量与置信度，避免论文里出现无出处的断言。
"""

import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from hermes.cross_domain.mental_model_trainer import MIN_RELIABLE_SAMPLES
from hermes.self_research.data_provenance import REAL, UNVERIFIED, worst
from hermes.self_research.research_data_collector import ResearchDataset
from hermes.self_research.statistical_analyzer import StatisticalAnalysis

DIRECTION_TEXT = {"up": "上升", "down": "下降", "flat": "持平"}


@dataclass
class Finding:
    finding_id: str
    statement: str
    evidence: str
    category: str
    metric: str
    value: float
    confidence: float
    significant: bool
    tags: List[str] = field(default_factory=list)
    data_fields: List[str] = field(default_factory=list)
    provenance: str = REAL
    sample_size: Optional[int] = None
    required_sample: Optional[int] = None
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "statement": self.statement,
            "evidence": self.evidence,
            "category": self.category,
            "metric": self.metric,
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 2),
            "significant": self.significant,
            "tags": self.tags,
            "data_fields": self.data_fields,
            "provenance": self.provenance,
            "sample_size": self.sample_size,
            "required_sample": self.required_sample,
            "caveats": self.caveats,
        }


class FindingExtractor:
    """按规则从统计结果中提炼关键发现。"""

    def __init__(
        self,
        alpha: float = 0.05,
        strategy_threshold: float = 0.05,
        weak_threshold: float = 0.01,
        similarity_threshold: float = 0.7,
        success_threshold: float = 0.8,
        max_findings: int = 12,
        min_samples: int = 3,
    ):
        self._alpha = alpha
        self._strategy_threshold = strategy_threshold
        self._weak_threshold = weak_threshold
        self._similarity_threshold = similarity_threshold
        self._success_threshold = success_threshold
        self._max_findings = max_findings
        self._min_samples = min_samples

    def extract(self, analysis: StatisticalAnalysis, dataset: ResearchDataset) -> List[Finding]:
        findings: List[Finding] = []
        findings.extend(self._trend_findings(analysis))
        findings.extend(self._strategy_findings(analysis))
        findings.extend(self._transfer_findings(analysis, dataset))
        findings.extend(self._evolution_findings(dataset))
        findings.extend(self._metacognition_findings(analysis))
        findings.extend(self._latency_findings(dataset))
        findings.extend(self._policy_findings(dataset))

        findings = findings[: self._max_findings]
        findings = [self._resolve_provenance(finding, dataset) for finding in findings]
        return [replace(finding, finding_id=f"F-{index:02d}") for index, finding in enumerate(findings, 1)]

    @staticmethod
    def _resolve_provenance(finding: Finding, dataset: ResearchDataset) -> Finding:
        """按数据字段的来源标签定下这条发现的来源（取最不可信的那个）。"""

        if not finding.data_fields:
            return finding if finding.provenance != REAL else replace(finding, provenance=UNVERIFIED)
        declared = [(dataset.provenance or {}).get(name) for name in finding.data_fields]
        resolved = worst(item for item in declared if item)
        return replace(finding, provenance=resolved)

    # ------------------------------------------------------------------ 规则

    def _trend_findings(self, analysis: StatisticalAnalysis) -> List[Finding]:
        trend = analysis.success_rate_trend
        if trend is None or trend.n < 3:
            return []

        direction = DIRECTION_TEXT.get(trend.direction, trend.direction)
        if trend.significant:
            statement = (
                f"总体成功率随快照推进显著{direction}：全程变化 {trend.total_change:+.1%}，"
                f"斜率 {trend.slope:+.4f}/快照"
            )
        else:
            statement = (
                f"总体成功率随快照推进的变化未达显著水平：全程变化 {trend.total_change:+.1%}，"
                f"斜率 {trend.slope:+.4f}/快照"
            )

        return [
            Finding(
                finding_id="",
                statement=statement,
                evidence=f"对 {trend.n} 个快照做最小二乘拟合，R²={trend.r_squared:.3f}，p={trend.p_value:.4f}",
                category="capability",
                metric="success_rate_trend",
                value=trend.slope,
                confidence=max(0.0, min(1.0, 1.0 - trend.p_value)),
                significant=trend.significant,
                tags=["trend", "success_rate"],
                data_fields=["snapshots"],
                sample_size=trend.n,
                required_sample=self._min_samples,
            )
        ]

    def _strategy_findings(self, analysis: StatisticalAnalysis) -> List[Finding]:
        findings: List[Finding] = []
        stats = analysis.strategy_stats
        if not stats:
            return findings

        for stat in stats:
            if stat.avg_benefit >= self._strategy_threshold:
                findings.append(
                    Finding(
                        finding_id="",
                        statement=(
                            f"策略 {stat.strategy_type} 显著有效：平均收益 {stat.avg_benefit:+.1%}，"
                            f"成功率 {stat.success_rate:.0%}"
                        ),
                        evidence=(
                            f"n={stat.instances}，收益中位数 {stat.median_benefit:+.1%}，"
                            f"标准差 {stat.stdev:.3f}，阈值 {self._strategy_threshold:.0%}"
                        ),
                        category="strategy",
                        metric="avg_benefit",
                        value=stat.avg_benefit,
                        confidence=0.9 if stat.instances >= 3 else 0.6,
                        significant=True,
                        tags=["strategy", stat.strategy_type],
                        data_fields=["strategies"],
                        sample_size=stat.instances,
                        required_sample=self._min_samples,
                    )
                )

        worst = stats[-1]
        if worst.avg_benefit <= self._weak_threshold:
            findings.append(
                Finding(
                    finding_id="",
                    statement=(
                        f"策略 {worst.strategy_type} 未观察到有效收益（平均 {worst.avg_benefit:+.1%}），" "建议停止投入"
                    ),
                    evidence=f"n={worst.instances}，中位数 {worst.median_benefit:+.1%}",
                    category="strategy",
                    metric="avg_benefit",
                    value=worst.avg_benefit,
                    confidence=0.6,
                    significant=False,
                    tags=["strategy", "negative"],
                    data_fields=["strategies"],
                    sample_size=worst.instances,
                    required_sample=self._min_samples,
                )
            )

        return findings

    def _transfer_findings(self, analysis: StatisticalAnalysis, dataset: ResearchDataset) -> List[Finding]:
        findings: List[Finding] = []
        pairs = dataset.similarity_pairs
        if not pairs:
            return findings

        strong = [
            pair
            for pair in pairs
            if float(pair.get("similarity", 0.0)) >= self._similarity_threshold
            and float(pair.get("success_rate", 0.0)) >= self._success_threshold
        ]

        if strong:
            best = max(strong, key=lambda p: float(p["similarity"]) * float(p["success_rate"]))
            findings.append(
                Finding(
                    finding_id="",
                    statement=(
                        f"相似度 {best['similarity']:.2f} 且成功率 {best['success_rate']:.0%} 的模式组合构成强迁移关系"
                        f"（{best['source_pattern']} → {best['target_pattern']}）"
                    ),
                    evidence=f"判据：相似度 ≥ {self._similarity_threshold:.2f} 且成功率 ≥ {self._success_threshold:.0%}",
                    category="transfer",
                    metric="transfer_strength",
                    value=float(best["similarity"]) * float(best["success_rate"]),
                    confidence=0.85,
                    significant=True,
                    tags=["transfer", "strong"],
                    data_fields=["similarity_pairs", "pattern_success_rates", "knowledge_graph"],
                    sample_size=len(pairs),
                    required_sample=self._min_samples,
                )
            )

        correlation = analysis.similarity_correlation
        if correlation is not None and correlation.n >= 3:
            findings.append(
                Finding(
                    finding_id="",
                    statement=(
                        f"相似度与迁移成功率的相关系数 r={correlation.r:+.3f}"
                        f"（{'显著' if correlation.significant else '不显著'}）"
                    ),
                    evidence=f"n={correlation.n}，p={correlation.p_value:.4f}",
                    category="transfer",
                    metric="similarity_correlation",
                    value=correlation.r,
                    confidence=max(0.0, min(1.0, 1.0 - correlation.p_value)),
                    significant=correlation.significant,
                    tags=["transfer", "correlation"],
                    data_fields=["similarity_pairs"],
                    sample_size=correlation.n,
                    required_sample=self._min_samples,
                )
            )
        elif correlation is not None:
            findings.append(
                Finding(
                    finding_id="",
                    statement=(f"当前仅有 {correlation.n} 组“相似度-成功率”配对观测，" "样本量不足以给出相关性结论"),
                    evidence="相关性检验至少需要 3 组配对观测",
                    category="transfer",
                    metric="similarity_pairs",
                    value=float(correlation.n),
                    confidence=0.4,
                    significant=False,
                    tags=["transfer", "limitation"],
                    data_fields=["similarity_pairs"],
                    sample_size=correlation.n,
                    required_sample=self._min_samples,
                    caveats=["样本量不足以支撑相关性结论"],
                )
            )

        return findings

    def _evolution_findings(self, dataset: ResearchDataset) -> List[Finding]:
        phases = dataset.phases
        if not phases:
            return []

        total_commits = sum(int(phase.get("total_commits", 0)) for phase in phases)
        # 带上阶段编号：不同阶段可能由同一种提交类型主导，只报名字会分不清
        names = "、".join(f"阶段{phase.get('stage_id')} {phase.get('name')}" for phase in phases)
        evidence = "；".join(f"{phase.get('name')}: {', '.join(phase.get('dominant_types') or [])}" for phase in phases)

        return [
            Finding(
                finding_id="",
                statement=f"自身仓库的演化可划分为 {len(phases)} 个阶段（{names}），共 {total_commits} 次提交",
                evidence=evidence,
                category="evolution",
                metric="phase_count",
                value=float(len(phases)),
                confidence=0.8,
                significant=False,
                tags=["evolution", "phases"],
                data_fields=["phases"],
                sample_size=total_commits,
                required_sample=self._min_samples,
            )
        ]

    def _metacognition_findings(self, analysis: StatisticalAnalysis) -> List[Finding]:
        findings: List[Finding] = []

        if analysis.mental_model_samples > 0 and not analysis.mental_model_reportable:
            return [
                Finding(
                    finding_id="",
                    statement=(
                        f"心智模型基于 {analysis.mental_model_samples} 个快照训练，"
                        "但样本量不足以支撑准确率结论，本文不报告该指标"
                    ),
                    evidence=(
                        "样本量低于 MentalModelTrainer 的可引用门槛" f"（{analysis.mental_model_note or '样本不足'}）"
                    ),
                    category="metacognition",
                    metric="prediction_accuracy",
                    value=0.0,
                    confidence=0.2,
                    significant=False,
                    tags=["metacognition", "prediction", "insufficient_data"],
                    data_fields=["mental_model"],
                    sample_size=analysis.mental_model_samples,
                    required_sample=MIN_RELIABLE_SAMPLES,
                    caveats=[analysis.mental_model_note or "样本量不足，准确率不具统计意义，不作为结论"],
                )
            ]

        if analysis.mental_model_samples > 0:
            findings.append(
                Finding(
                    finding_id="",
                    statement=(
                        f"心智模型基于 {analysis.mental_model_samples} 个快照训练，"
                        f"对未来状态的预测准确率为 {analysis.mental_model_accuracy:.0%}"
                    ),
                    evidence="预测准确率来自 MentalModelTrainer 的留出评估",
                    category="metacognition",
                    metric="prediction_accuracy",
                    value=analysis.mental_model_accuracy,
                    confidence=0.7,
                    significant=analysis.mental_model_accuracy > 0.6,
                    tags=["metacognition", "prediction"],
                    data_fields=["mental_model"],
                    sample_size=analysis.mental_model_samples,
                    required_sample=self._min_samples,
                )
            )

        if analysis.corrections > 0:
            findings.append(
                Finding(
                    finding_id="",
                    statement=f"自修复机制在 {analysis.corrections} 个快照上被触发并生效",
                    evidence="统计快照元数据中标记为 corrected 的记录数",
                    category="self_repair",
                    metric="corrections",
                    value=float(analysis.corrections),
                    confidence=0.75,
                    significant=True,
                    tags=["self_repair"],
                    data_fields=["snapshots"],
                )
            )

        return findings

    def _latency_findings(self, dataset: ResearchDataset) -> List[Finding]:
        pending: Dict[str, List[int]] = {}
        gaps: List[int] = []

        for index, event in enumerate(dataset.events):
            metadata = event.get("metadata") or {}
            target = metadata.get("target") or metadata.get("pattern")
            if not target:
                continue
            if event.get("event_type") == "evolution_target_selected":
                pending.setdefault(str(target), []).append(index)
            elif event.get("event_type") == "external_insight_applied" and pending.get(str(target)):
                gaps.append(index - pending[str(target)].pop(0))

        if not gaps:
            return []

        median_gap = statistics.median(gaps)
        return [
            Finding(
                finding_id="",
                statement=(f"跨域迁移从目标选定到外部洞察落地，中位间隔约 {median_gap:.0f} 个事件步（估计值）"),
                evidence=f"n={len(gaps)}，最小 {min(gaps)}，最大 {max(gaps)}；以事件日志位置为计时单位",
                category="transfer",
                metric="migration_latency_events",
                value=float(median_gap),
                confidence=0.5,
                significant=False,
                tags=["transfer", "latency", "estimate"],
                data_fields=["events"],
                sample_size=len(gaps),
                required_sample=5,
                caveats=["以事件日志位置为计时单位，属于估计值而非实测延迟"],
            )
        ]

    def _policy_findings(self, dataset: ResearchDataset) -> List[Finding]:
        weights = dataset.policy.get("pattern_weights") or {}
        if not weights:
            return []

        top = sorted(weights.items(), key=lambda item: (-float(item[1]), str(item[0])))[:3]
        top_text = "、".join(f"{name}({float(weight):.2f})" for name, weight in top)

        return [
            Finding(
                finding_id="",
                statement=f"进化策略累计更新 {dataset.policy.get('update_count', 0)} 次，当前权重最高的模式为 {top_text}",
                evidence=f"采样模式总数 {len(weights)}，变异率 {dataset.policy.get('mutation_rate', 0)}",
                category="policy",
                metric="policy_update_count",
                value=float(dataset.policy.get("update_count", 0) or 0),
                confidence=0.65,
                significant=False,
                tags=["policy"],
                data_fields=["policy"],
            )
        ]
