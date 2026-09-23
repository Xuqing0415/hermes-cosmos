"""对每条审稿意见生成作者回应（接受 / 部分接受 / 需补充实验 / 反驳）。

有两条硬规则，写在代码里也写在测试里：

1. **只有次要（MINOR）意见允许反驳**。致命/重大意见一律不接受“反驳”这个选项——
   一个能随手驳回自己致命缺陷的系统，等于没有审稿。
2. **任何回应都不能把结论改得更好**。反驳只影响措辞，并且**不免掉**该条意见对应的
   论文修订：证据等级的修订权只属于 `PaperRevisionEngine`，它按排序结果执行动作，
   且只会往下调——所以一条被反驳的次要意见仍可能把结论的证据等级上限压到 B。

回应措辞是规则生成的，不调用任何模型，因此可复现、可审计。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes.self_research.adversarial_reviewer import (
    CORRELATION_CAUSATION,
    FATAL,
    MINOR,
    MULTIPLE_COMPARISONS,
    NO_CONTROL_GROUP,
    OVERGENERALIZATION,
    SAMPLE_SIZE,
    SURVIVORSHIP,
    WEAK_EVIDENCE,
    Attack,
)

ACCEPT = "accept"
PARTIAL = "partial"
NEEDS_EVIDENCE = "needs_evidence"
REBUT = "rebut"

STANCE_LABELS = {
    ACCEPT: "接受",
    PARTIAL: "部分接受",
    NEEDS_EVIDENCE: "需补充实验",
    REBUT: "反驳",
}

#: 只有次要意见可以被反驳
REBUTTABLE_SEVERITIES = (MINOR,)

#: 回应之后论文必须做的动作
REVISION_WITHDRAW = "withdraw"
REVISION_DOWNGRADE = "downgrade"
REVISION_REWORD = "reword"
REVISION_SCOPE = "scope"
REVISION_DECLARE_LIMIT = "declare_limit"


@dataclass
class Rebuttal:
    attack_id: str
    finding_id: str
    severity: str
    stance: str
    response: str
    revision: str

    @property
    def stance_label(self) -> str:
        return STANCE_LABELS.get(self.stance, self.stance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "finding_id": self.finding_id,
            "severity": self.severity,
            "stance": self.stance,
            "stance_label": self.stance_label,
            "response": self.response,
            "revision": self.revision,
        }


class RebuttalGenerator:
    def generate_all(self, attacks: List[Attack]) -> List[Rebuttal]:
        return [self.generate(attack) for attack in attacks]

    def generate(self, attack: Attack) -> Rebuttal:
        stance, response, revision = self._respond(attack)
        if attack.severity not in REBUTTABLE_SEVERITIES and stance == REBUT:
            # 兜底：致命/重大意见不允许被反驳
            stance, response, revision = (
                ACCEPT,
                f"该意见无法反驳，接受修改：{attack.suggested_fix}",
                REVISION_DOWNGRADE,
            )
        return Rebuttal(
            attack_id=attack.attack_id,
            finding_id=attack.finding_id,
            severity=attack.severity,
            stance=stance,
            response=response,
            revision=revision,
        )

    # ------------------------------------------------------------------ 规则

    def _respond(self, attack: Attack) -> Any:
        handler = {
            SAMPLE_SIZE: self._sample_size,
            WEAK_EVIDENCE: self._weak_evidence,
            NO_CONTROL_GROUP: self._no_control_group,
            CORRELATION_CAUSATION: self._causation,
            SURVIVORSHIP: self._survivorship,
            MULTIPLE_COMPARISONS: self._multiple_comparisons,
            OVERGENERALIZATION: self._overgeneralization,
        }.get(attack.attack_type)
        if handler is None:
            return (
                PARTIAL,
                f"接受该意见中可验证的部分：{attack.suggested_fix}",
                REVISION_REWORD,
            )
        return handler(attack)

    @staticmethod
    def _sample_size(attack: Attack):
        if attack.severity == FATAL:
            return (
                ACCEPT,
                f"接受。该结论只有单一数据点（{attack.evidence}），无法支撑任何推广，予以撤回。",
                REVISION_WITHDRAW,
            )
        return (
            ACCEPT,
            f"接受。样本量 {attack.evidence} 不足以支撑主要发现，将该结论降级为初步观察。",
            REVISION_DOWNGRADE,
        )

    @staticmethod
    def _weak_evidence(attack: Attack):
        return (
            ACCEPT,
            f"接受。该结论的证据等级为 {attack.evidence}，按等级重述为初步结论。",
            REVISION_DOWNGRADE,
        )

    @staticmethod
    def _no_control_group(attack: Attack):
        return (
            NEEDS_EVIDENCE,
            "接受该批评：没有对照组就无法把收益归因于策略。"
            "本版先把因果措辞改为“伴随/与…一致”，并把缺少对照组写进局限性；"
            "补充对照实验后才能给出因果结论。",
            REVISION_REWORD,
        )

    @staticmethod
    def _causation(attack: Attack):
        return (
            PARTIAL,
            "部分接受。数据确实只能支持相关性，措辞改为“与…一致”，" "但结论方向保留（相关性本身仍是有价值的信息）。",
            REVISION_REWORD,
        )

    @staticmethod
    def _survivorship(attack: Attack):
        return (
            NEEDS_EVIDENCE,
            f"接受该批评（{attack.evidence}）。失败样本没有被采集，"
            "本版明确声明该局限，并标注“有效”结论仅适用于被记录的尝试；"
            "补齐失败记录后才能给出无偏估计。",
            REVISION_DECLARE_LIMIT,
        )

    @staticmethod
    def _multiple_comparisons(attack: Attack):
        return (
            ACCEPT,
            f"接受。共 {attack.evidence}，按校正后标准该结论不再显著，" "改为报告校正阈值并去掉“显著”字样。",
            REVISION_DOWNGRADE,
        )

    @staticmethod
    def _overgeneralization(attack: Attack):
        if attack.severity == MINOR:
            return (
                REBUT,
                "部分反驳：本文并未声称该结论可外推到其他仓库，"
                "标题与摘要都已限定为“本系统的自身演化”，此处只是措辞不够明确。",
                REVISION_SCOPE,
            )
        return (
            PARTIAL,
            f"部分接受（{attack.evidence}）。把结论明确限定在已有证据的范围内，"
            "并删去一般性措辞；但保留对已有领域的描述性结论。",
            REVISION_SCOPE,
        )

    @staticmethod
    def stance_of(rebuttals: List[Rebuttal], attack_id: str) -> Optional[str]:
        for item in rebuttals:
            if item.attack_id == attack_id:
                return item.stance
        return None
