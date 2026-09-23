"""对每条审稿意见生成作者回应（接受 / 部分接受 / 需补充实验 / 反驳）。

有两条硬规则，写在代码里也写在测试里：

1. **只有次要（MINOR）意见允许反驳**。致命/重大意见一律不接受“反驳”这个选项——
  一个能随手驳回自己致命缺陷的系统，等于没有审稿。
2. **反驳是论证，不是降级开关**。反驳必须给出可核对的事实（本模块只核对“论文是否
   真的已经限定过适用范围”这类能对着正文验证的理由），因此每条回应都带一个
   `strength`：
   * `strong`（理由可核对）——维持原始证据等级，但论文里必须附上反驳理由；
   * `weak`（理由无法核对）——按作者强辩处理，证据等级上限压到 C。
   没有反驳（`accept` / `partial` / `needs_evidence`）时等级上限仍由该条意见的
   排序动作决定，`PaperRevisionEngine` 只会往下调，绝不会因为回应而升上去。

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
#: 反驳理由成立：不降级，但论文里必须附上理由
REVISION_MAINTAIN = "maintain"

#: 反驳理由的可核对程度
STRENGTH_NONE = "none"
STRENGTH_STRONG = "strong"
STRENGTH_WEAK = "weak"

STRENGTH_LABELS = {
    STRENGTH_NONE: "不适用",
    STRENGTH_STRONG: "理由可核对",
    STRENGTH_WEAK: "理由无法核对",
}

#: 审计章节的标题：核对反驳理由时必须把审计章节排除，否则反驳会“自我作证”
AUDIT_SECTION_MARKER = "审稿意见与作者回应"

#: 能证明“适用范围已被限定”的措辞（用于核对过度泛化类反驳）
SCOPE_LIMIT_MARKERS = ("本系统", "本仓库", "自身演化", "不声称", "仅限", "不超出")


@dataclass
class Rebuttal:
    attack_id: str
    finding_id: str
    severity: str
    stance: str
    response: str
    revision: str
    strength: str = STRENGTH_NONE

    @property
    def stance_label(self) -> str:
        return STANCE_LABELS.get(self.stance, self.stance)

    @property
    def strength_label(self) -> str:
        return STRENGTH_LABELS.get(self.strength, self.strength)

    @property
    def stance_with_strength(self) -> str:
        """表格/终端里显示的回应标签：反驳会带上理由是否可核对。"""

        if self.stance != REBUT:
            return self.stance_label
        return f"{self.stance_label}（{self.strength_label}）"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "finding_id": self.finding_id,
            "severity": self.severity,
            "stance": self.stance,
            "stance_label": self.stance_label,
            "strength": self.strength,
            "strength_label": self.strength_label,
            "response": self.response,
            "revision": self.revision,
        }


class RebuttalGenerator:
    def generate_all(self, attacks: List[Attack], paper_text: str = "") -> List[Rebuttal]:
        return [self.generate(attack, paper_text=paper_text) for attack in attacks]

    def generate(self, attack: Attack, paper_text: str = "") -> Rebuttal:
        stance, response, revision = self._respond(attack)
        if attack.severity not in REBUTTABLE_SEVERITIES and stance == REBUT:
            # 兜底：致命/重大意见不允许被反驳
            stance, response, revision = (
                ACCEPT,
                f"该意见无法反驳，接受修改：{attack.suggested_fix}",
                REVISION_DOWNGRADE,
            )
        strength = self._strength(attack, stance, paper_text)
        if stance == REBUT:
            # 反驳是否降级，取决于理由能不能被核对，而不是“有没有反驳”
            revision = REVISION_MAINTAIN if strength == STRENGTH_STRONG else REVISION_DOWNGRADE
        return Rebuttal(
            attack_id=attack.attack_id,
            finding_id=attack.finding_id,
            severity=attack.severity,
            stance=stance,
            response=response,
            revision=revision,
            strength=strength,
        )

    # ------------------------------------------------------------------ 规则

    def _strength(self, attack: Attack, stance: str, paper_text: str) -> str:
        """核对反驳理由。核对不了就按“作者强辩”处理，这是唯一安全的默认值。"""

        if stance != REBUT:
            return STRENGTH_NONE
        verifiers = {OVERGENERALIZATION: self._verify_scope_claim}
        verifier = verifiers.get(attack.attack_type)
        if verifier is None:
            return STRENGTH_WEAK
        return STRENGTH_STRONG if verifier(paper_text) else STRENGTH_WEAK

    @staticmethod
    def _verify_scope_claim(paper_text: str) -> bool:
        """核对“本文已限定适用范围”这一理由，只看审计章节之前的正文。"""

        body = str(paper_text or "").split(AUDIT_SECTION_MARKER)[0]
        return any(marker in body for marker in SCOPE_LIMIT_MARKERS)

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
