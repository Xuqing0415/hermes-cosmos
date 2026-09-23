"""Self-Researcher 分析层：描述性统计、趋势检验、组间对比与相关性分析。

全部统计量只用标准库（`math` / `statistics`）实现，不依赖 numpy：
CI 环境与无科学计算依赖的机器上得到完全一致的结果。
显著性检验使用 Student-t 分布（经由正则化不完全 beta 函数求 p 值）。
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hermes.self_research.research_data_collector import ResearchDataset


def _betacf(a: float, b: float, x: float, max_iterations: int = 200, epsilon: float = 3e-9) -> float:
    """不完全 beta 函数的连分式展开（Lentz 算法）。"""

    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 `I_x(a, b)`。"""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_statistic: float, degrees_of_freedom: float) -> float:
    """双侧 Student-t 检验的 p 值。"""

    if degrees_of_freedom <= 0 or math.isnan(t_statistic):
        return 1.0
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic)
    return max(0.0, min(1.0, _betai(degrees_of_freedom / 2.0, 0.5, x)))


def describe(values: Sequence[float]) -> "DescriptiveStats":
    """序列的描述性统计。"""

    data = [float(v) for v in values]
    if not data:
        return DescriptiveStats()

    return DescriptiveStats(
        count=len(data),
        mean=statistics.fmean(data),
        stdev=statistics.stdev(data) if len(data) >= 2 else 0.0,
        minimum=min(data),
        maximum=max(data),
        first=data[0],
        last=data[-1],
        delta=data[-1] - data[0],
    )


def linear_trend(values: Sequence[float], alpha: float = 0.05) -> "TrendResult":
    """对序列做最小二乘线性拟合，并检验斜率是否显著异于 0。"""

    y = [float(v) for v in values]
    n = len(y)

    if n < 3:
        return TrendResult(
            slope=0.0,
            intercept=y[-1] if y else 0.0,
            r_squared=0.0,
            p_value=1.0,
            significant=False,
            direction="flat",
            n=n,
            total_change=(y[-1] - y[0]) if n >= 2 else 0.0,
        )

    x = list(range(n))
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)

    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = y_mean - slope * x_mean

    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sse = sum(residual * residual for residual in residuals)
    sst = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1.0 - sse / sst if sst else 0.0

    if sse <= 1e-15:
        p_value = 0.0 if abs(slope) > 0 else 1.0
    else:
        standard_error = math.sqrt((sse / (n - 2)) / sxx) if sxx else 0.0
        if standard_error <= 0:
            p_value = 1.0
        else:
            p_value = student_t_two_sided_p(slope / standard_error, n - 2)

    direction = "up" if slope > 0 else ("down" if slope < 0 else "flat")
    return TrendResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        significant=p_value < alpha,
        direction=direction,
        n=n,
        total_change=y[-1] - y[0],
    )


def pearson_correlation(xs: Sequence[float], ys: Sequence[float], alpha: float = 0.05) -> "CorrelationResult":
    """Pearson 相关系数及其显著性。"""

    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 3:
        return CorrelationResult(n=n, r=0.0, p_value=1.0, significant=False)

    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    syy = sum((yi - y_mean) ** 2 for yi in y)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))

    if sxx <= 0 or syy <= 0:
        return CorrelationResult(n=n, r=0.0, p_value=1.0, significant=False)

    r = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_statistic = r * math.sqrt((n - 2) / (1.0 - r * r))
        p_value = student_t_two_sided_p(t_statistic, n - 2)

    return CorrelationResult(n=n, r=r, p_value=p_value, significant=p_value < alpha)


def welch_t_test(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[float, float]:
    """Welch 双侧 t 检验，返回 `(t, p)`；样本不足时返回 `(0.0, 1.0)`。"""

    a = [float(v) for v in sample_a]
    b = [float(v) for v in sample_b]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0

    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    n_a, n_b = len(a), len(b)

    denominator = var_a / n_a + var_b / n_b
    if denominator <= 0:
        return 0.0, 1.0

    t_statistic = (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(denominator)
    degrees_of_freedom = denominator**2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
    return t_statistic, student_t_two_sided_p(t_statistic, degrees_of_freedom)


@dataclass
class DescriptiveStats:
    count: int = 0
    mean: float = 0.0
    stdev: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    first: float = 0.0
    last: float = 0.0
    delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "stdev": round(self.stdev, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "first": round(self.first, 4),
            "last": round(self.last, 4),
            "delta": round(self.delta, 4),
        }


@dataclass
class TrendResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    significant: bool
    direction: str
    n: int
    total_change: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "r_squared": round(self.r_squared, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "direction": self.direction,
            "n": self.n,
            "total_change": round(self.total_change, 4),
        }


@dataclass
class CorrelationResult:
    n: int
    r: float
    p_value: float
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "r": round(self.r, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


@dataclass
class StrategyStat:
    strategy_type: str
    instances: int
    avg_benefit: float
    median_benefit: float
    stdev: float
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "instances": self.instances,
            "avg_benefit": round(self.avg_benefit, 4),
            "median_benefit": round(self.median_benefit, 4),
            "stdev": round(self.stdev, 4),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class ComparisonResult:
    label: str
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_value: float
    significant: bool
    basis: str = "snapshots"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "basis": self.basis,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "difference": round(self.difference, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


@dataclass
class StatisticalAnalysis:
    success_rate: DescriptiveStats = field(default_factory=DescriptiveStats)
    cross_domain_success: DescriptiveStats = field(default_factory=DescriptiveStats)
    pattern_coverage: DescriptiveStats = field(default_factory=DescriptiveStats)
    success_rate_trend: Optional[TrendResult] = None
    cross_domain_trend: Optional[TrendResult] = None
    coverage_trend: Optional[TrendResult] = None
    strategy_stats: List[StrategyStat] = field(default_factory=list)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    similarity_correlation: Optional[CorrelationResult] = None
    mental_model_accuracy: float = 0.0
    mental_model_samples: int = 0
    #: True 表示 mental_model_accuracy 不是实测值（样本不足，未做留出评测）
    mental_model_estimated: bool = False
    #: False 表示准确率不具统计意义（样本量不足），论文中不得引用
    mental_model_reliable: bool = False
    mental_model_eval_samples: int = 0
    mental_model_note: str = ""
    corrections: int = 0

    @property
    def mental_model_reportable(self) -> bool:
        """准确率是否可以写进论文：必须是实测值、样本充足、且确实训练过。"""

        return self.mental_model_samples > 0 and not self.mental_model_estimated and self.mental_model_reliable

    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "success_rate": self.success_rate.to_dict(),
            "cross_domain_success": self.cross_domain_success.to_dict(),
            "pattern_coverage": self.pattern_coverage.to_dict(),
            "success_rate_trend": self.success_rate_trend.to_dict() if self.success_rate_trend else None,
            "cross_domain_trend": self.cross_domain_trend.to_dict() if self.cross_domain_trend else None,
            "coverage_trend": self.coverage_trend.to_dict() if self.coverage_trend else None,
            "strategy_stats": [s.to_dict() for s in self.strategy_stats],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "similarity_correlation": self.similarity_correlation.to_dict() if self.similarity_correlation else None,
            "mental_model_accuracy": round(self.mental_model_accuracy, 4),
            "mental_model_samples": self.mental_model_samples,
            "mental_model_estimated": self.mental_model_estimated,
            "mental_model_reliable": self.mental_model_reliable,
            "mental_model_eval_samples": self.mental_model_eval_samples,
            "mental_model_note": self.mental_model_note,
            "mental_model_reportable": self.mental_model_reportable,
            "corrections": self.corrections,
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
        }


class StatisticalAnalyzer:
    """把原始数据集折算成可写进论文的统计结论。"""

    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha

    def analyze(self, dataset: ResearchDataset) -> StatisticalAnalysis:
        success = [float(s.get("success_rate", 0.0)) for s in dataset.snapshots]
        cross = [float(s.get("cross_domain_success", 0.0)) for s in dataset.snapshots]
        coverage = [float(s.get("pattern_coverage", 0.0)) for s in dataset.snapshots]

        strategy_stats = self._strategy_stats(dataset.strategies)

        return StatisticalAnalysis(
            success_rate=describe(success),
            cross_domain_success=describe(cross),
            pattern_coverage=describe(coverage),
            success_rate_trend=linear_trend(success, self._alpha) if success else None,
            cross_domain_trend=linear_trend(cross, self._alpha) if cross else None,
            coverage_trend=linear_trend(coverage, self._alpha) if coverage else None,
            strategy_stats=strategy_stats,
            comparisons=self._build_comparisons(success, dataset.strategies, strategy_stats),
            similarity_correlation=pearson_correlation(
                [p.get("similarity", 0.0) for p in dataset.similarity_pairs],
                [p.get("success_rate", 0.0) for p in dataset.similarity_pairs],
                self._alpha,
            ),
            mental_model_accuracy=float(dataset.mental_model.get("accuracy", 0.0) or 0.0),
            mental_model_samples=int(dataset.mental_model.get("training_samples", 0) or 0),
            mental_model_estimated=bool(dataset.mental_model.get("estimated")),
            # 未声明 reliable 的模型一律按“不可引用”处理
            mental_model_reliable=bool(dataset.mental_model.get("reliable", False)),
            mental_model_eval_samples=int(dataset.mental_model.get("evaluation_samples", 0) or 0),
            mental_model_note=str(dataset.mental_model.get("note") or ""),
            corrections=sum(1 for s in dataset.snapshots if (s.get("metadata") or {}).get("corrected")),
            best_strategy=strategy_stats[0].strategy_type if strategy_stats else None,
            worst_strategy=strategy_stats[-1].strategy_type if strategy_stats else None,
            alpha=self._alpha,
        )

    def _strategy_stats(self, strategies: List[Dict[str, Any]]) -> List[StrategyStat]:
        groups: Dict[str, List[float]] = {}
        for observation in strategies:
            name = str(observation.get("strategy_type") or "unknown")
            groups.setdefault(name, []).append(float(observation.get("benefit", 0.0)))

        stats: List[StrategyStat] = []
        for name, benefits in groups.items():
            positive = sum(1 for b in benefits if b > 0)
            stats.append(
                StrategyStat(
                    strategy_type=name,
                    instances=len(benefits),
                    avg_benefit=statistics.fmean(benefits),
                    median_benefit=statistics.median(benefits),
                    stdev=statistics.stdev(benefits) if len(benefits) >= 2 else 0.0,
                    success_rate=positive / len(benefits),
                )
            )

        stats.sort(key=lambda s: (-s.avg_benefit, -s.instances, s.strategy_type))
        return stats

    def _build_comparisons(
        self,
        success: List[float],
        strategies: List[Dict[str, Any]],
        strategy_stats: List[StrategyStat],
    ) -> List[ComparisonResult]:
        comparisons: List[ComparisonResult] = []

        # 快照前后期对比：快照序列的早半段 vs 晚半段。
        # 注意：这是按快照序号切分的，与 git 演化阶段无关，命名和基都必须如实说明。
        if len(success) >= 6:
            half = len(success) // 2
            t_statistic, p_value = welch_t_test(success[half:], success[:half])
            comparisons.append(
                ComparisonResult(
                    label="快照前后期对比（早半段 vs 晚半段）",
                    group_a="快照后半段",
                    group_b="快照前半段",
                    mean_a=statistics.fmean(success[half:]),
                    mean_b=statistics.fmean(success[:half]),
                    difference=statistics.fmean(success[half:]) - statistics.fmean(success[:half]),
                    t_statistic=t_statistic,
                    p_value=p_value,
                    significant=p_value < self._alpha,
                    basis="snapshots",
                )
            )

        # 策略对比：样本量最大的两个策略
        ranked = sorted(strategy_stats, key=lambda s: (-s.instances, s.strategy_type))
        if len(ranked) >= 2:
            first, second = ranked[0], ranked[1]
            first_values = self._benefits_of(strategies, first.strategy_type)
            second_values = self._benefits_of(strategies, second.strategy_type)
            t_statistic, p_value = welch_t_test(first_values, second_values)
            comparisons.append(
                ComparisonResult(
                    label="策略对比（样本量前二）",
                    group_a=first.strategy_type,
                    group_b=second.strategy_type,
                    mean_a=first.avg_benefit,
                    mean_b=second.avg_benefit,
                    difference=first.avg_benefit - second.avg_benefit,
                    t_statistic=t_statistic,
                    p_value=p_value,
                    significant=p_value < self._alpha,
                    basis="strategy_observations",
                )
            )

        return comparisons

    @staticmethod
    def _benefits_of(strategies: List[Dict[str, Any]], strategy_type: str) -> List[float]:
        return [float(o.get("benefit", 0.0)) for o in strategies if str(o.get("strategy_type")) == strategy_type]
