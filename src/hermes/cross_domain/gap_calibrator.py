from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .evolution_compare_engine import ComparisonReport, StageComparison
from .mental_model_trainer import MentalModelTrainer


@dataclass
class CalibrationSuggestion:
    calibration_id: str
    target: str
    current_value: float
    adjusted_value: float
    expected_accuracy_gain: float
    description: str
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "target": self.target,
            "current_value": round(self.current_value, 2),
            "adjusted_value": round(self.adjusted_value, 2),
            "expected_accuracy_gain": round(self.expected_accuracy_gain, 2),
            "description": self.description,
            "priority": self.priority,
        }


BIAS_CALIBRATIONS = {
    "conservative": [
        {
            "target": "cross_domain_success_weight",
            "description": "心智模型系统性地低估了跨领域迁移的延迟效应。"
                           "将'跨域迁移成熟期'的预测窗口向后调整 1 个阶段",
            "delta": 0.15,
            "gain": 0.10,
        },
        {
            "target": "pattern_maturity_threshold",
            "description": "模式成熟判定阈值过低，导致过早收敛。"
                           "将置信度阈值提高以延长学习窗口",
            "delta": 0.10,
            "gain": 0.08,
        },
    ],
    "optimistic": [
        {
            "target": "initial_benefit_estimate",
            "description": "心智模型系统性地高估了策略的初期效果。"
                           "将初始收益预期下调以更贴近实际曲线",
            "delta": -0.10,
            "gain": 0.12,
        },
        {
            "target": "decay_factor",
            "description": "策略收益衰减速度被低估。增加衰减因子",
            "delta": 0.05,
            "gain": 0.08,
        },
    ],
    "mixed": [
        {
            "target": "success_rate_learning_rate",
            "description": "心智模型在不同阶段的准确率波动较大。"
                           "调整学习率以平衡各阶段的预测精度",
            "delta": 0.05,
            "gain": 0.15,
        },
        {
            "target": "feature_importance_weights",
            "description": "特征重要性分布需要重新校准。"
                           "增加历史实际数据的权重，减少模拟数据的权重",
            "delta": 0.10,
            "gain": 0.12,
        },
    ],
    "balanced": [
        {
            "target": "fine_tune",
            "description": "模型已较为准确，仅需微调边界条件。"
                           "对离群点进行额外采样",
            "delta": 0.02,
            "gain": 0.05,
        },
    ],
}


class GapCalibrator:
    def __init__(self, trainer: Optional[MentalModelTrainer] = None):
        self._trainer = trainer
        self._suggestions: List[CalibrationSuggestion] = []

    def analyze(self, comparison_report: ComparisonReport) -> List[CalibrationSuggestion]:
        self._suggestions.clear()

        bias = comparison_report.model_bias
        calibrations = BIAS_CALIBRATIONS.get(bias, BIAS_CALIBRATIONS["balanced"])

        for cal in calibrations:
            self._suggestions.append(CalibrationSuggestion(
                calibration_id=f"cal-{cal['target']}",
                target=cal["target"],
                current_value=0.5,
                adjusted_value=0.5 + cal["delta"],
                expected_accuracy_gain=cal["gain"],
                description=cal["description"],
                priority=calibrations.index(cal) + 1,
            ))

        # Add stage-specific suggestions
        for comp in comparison_report.comparisons:
            if comp.over_under == "underestimated" and comp.delta < -0.1:
                self._suggestions.append(CalibrationSuggestion(
                    calibration_id=f"cal-stage-{comp.stage_id}",
                    target=f"stage_{comp.stage_id}_prediction",
                    current_value=comp.predicted_success_rate,
                    adjusted_value=comp.actual_success_rate,
                    expected_accuracy_gain=0.08,
                    description=f"调整阶段 {comp.stage_id} ({comp.stage_name}) 的预测基线",
                    priority=len(self._suggestions) + 1,
                ))

        self._suggestions.sort(key=lambda s: (s.priority, s.expected_accuracy_gain), reverse=True)
        return self._suggestions

    def get_suggestions(self) -> List[CalibrationSuggestion]:
        return self._suggestions