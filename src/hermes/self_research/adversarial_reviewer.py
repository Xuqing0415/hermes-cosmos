"""对抗性审计：以审稿人的身份攻击系统自己写出的论文。

研究完整性引擎回答的是“这些数据是不是真的”。本模块回答另一个问题：
“就算数据是真的，这些结论站得住吗？” 审计器只负责找漏洞，
不负责让论文看起来更好——它没有任何“往上抬分”的路径。

审计依据的事实全部来自论文目录下已经落盘的产物
（`dataset.json` / `analysis.json` / `integrity.json` / `paper.*`），
不去解析正文来猜数字。缺哪个产物就显式记进 `AuditReport.skipped`，
并且把“确实跑过、没发现问题”的检查记进 `checks_run`——
静默跳过和静默通过都会被当成审计失职。
"""

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

FATAL = "FATAL"
MAJOR = "MAJOR"
MINOR = "MINOR"

SEVERITY_ORDER = (FATAL, MAJOR, MINOR)
SEVERITY_LABELS = {FATAL: "致命", MAJOR: "重大", MINOR: "次要"}

# ------------------------------------------------------------------ 攻击类型

SAMPLE_SIZE = "sample_size"
WEAK_EVIDENCE = "weak_evidence"
NO_CONTROL_GROUP = "no_control_group"
CORRELATION_CAUSATION = "correlation_causation"
SURVIVORSHIP = "survivorship"
MULTIPLE_COMPARISONS = "multiple_comparisons"
OVERGENERALIZATION = "overgeneralization"

ATTACK_LABELS = {
    SAMPLE_SIZE: "样本量攻击",
    WEAK_EVIDENCE: "证据等级攻击",
    NO_CONTROL_GROUP: "无对照组",
    CORRELATION_CAUSATION: "相关当因果",
    SURVIVORSHIP: "幸存者偏差",
    MULTIPLE_COMPARISONS: "多重比较",
    OVERGENERALIZATION: "过度泛化",
}

DEFAULT_MIN_PRIMARY_SAMPLE = 10
DEFAULT_MIN_DOMAINS = 3
DEFAULT_MIN_REPOSITORIES = 2

#: 哪些指标算“主张了某种效应”，值得审稿人开火
CLAIM_METRICS = (
    "strategy_benefit",
    "transfer_strength",
    "transfer_latency",
    "prediction_accuracy",
    "self_repair",
    "policy_update",
)

#: 已经自我否定的结论不再重复攻击（避免噪声）
INSUFFICIENCY_MARKERS = ("不足以", "不报告", "未达显著", "样本不足", "无法判断", "不具统计意义")

#: 发现级因果措辞（把观察说成作用）
CAUSAL_MARKERS = ("导致", "引起", "使得", "带来", "有效", "驱动", "归因")

#: 相关性类指标，出现因果措辞就是“相关当因果”
CORRELATIONAL_METRICS = ("transfer_strength", "similarity_success_correlation", "similarity_pairs")

#: 正文里相关性 + 因果措辞同时出现才算命中（避免误报）
CORRELATION_TERMS = ("相关性", "相关系数", "相似度")
PAPER_CAUSAL_MARKERS = ("导致", "引起", "使得", "带来", "证明了", "说明了")

SENTENCE_SPLIT_RE = re.compile(r"[。；\n]")


@dataclass
class Attack:
    attack_id: str
    attack_type: str
    severity: str
    target: str
    statement: str
    evidence: str
    suggested_fix: str
    finding_id: str = ""
    category: str = "other"

    @property
    def label(self) -> str:
        return ATTACK_LABELS.get(self.attack_type, self.attack_type)

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS.get(self.severity, self.severity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "attack_type": self.attack_type,
            "attack_label": self.label,
            "severity": self.severity,
            "severity_label": self.severity_label,
            "target": self.target,
            "finding_id": self.finding_id,
            "category": self.category,
            "statement": self.statement,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class AuditReport:
    attacks: List[Attack] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    context_available: Dict[str, bool] = field(default_factory=dict)
    confidence: Optional[float] = None
    self_limited: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.attacks

    @property
    def severity_counts(self) -> Dict[str, int]:
        counts = dict.fromkeys(SEVERITY_ORDER, 0)
        for attack in self.attacks:
            counts[attack.severity] = counts.get(attack.severity, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "attack_count": len(self.attacks),
            "severity_counts": self.severity_counts,
            "attacks": [attack.to_dict() for attack in self.attacks],
            "checks_run": self.checks_run,
            "skipped": self.skipped,
            "self_limited": self.self_limited,
            "context_available": self.context_available,
            "baseline_confidence": self.confidence,
        }


@dataclass
class ReviewContext:
    """审计所需的全部已落盘事实。"""

    directory: str = ""
    paper_path: str = ""
    paper_text: str = ""
    dataset: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    integrity: Dict[str, Any] = field(default_factory=dict)
    available: Dict[str, bool] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    @property
    def grades(self) -> Dict[str, Dict[str, Any]]:
        return {str(item.get("finding_id")): item for item in self.integrity.get("grades") or []}

    @property
    def confidence(self) -> Optional[float]:
        payload = self.integrity.get("confidence") or {}
        score = payload.get("score")
        return float(score) if isinstance(score, (int, float)) else None

    @property
    def alpha(self) -> float:
        try:
            return float(self.analysis.get("alpha") or 0.05)
        except (TypeError, ValueError):
            return 0.05

    def finding(self, finding_id: str) -> Dict[str, Any]:
        for item in self.findings:
            if str(item.get("finding_id")) == finding_id:
                return item
        return {}


class ReviewContextLoader:
    """从论文目录加载审计事实；缺失的产物如实记录，不做任何推断。"""

    ARTIFACTS = (
        ("dataset", "dataset.json"),
        ("analysis", "analysis.json"),
        ("integrity", "integrity.json"),
    )
    PAPER_CANDIDATES = ("paper.md", "paper.tex")

    def load(self, path: str) -> ReviewContext:
        directory, paper_path = self._resolve(path)
        context = ReviewContext(directory=directory, paper_path=paper_path)

        for key, filename in self.ARTIFACTS:
            payload = self._read_json(os.path.join(directory, filename))
            context.available[key] = payload is not None
            if payload is None:
                context.missing.append(filename)
                continue
            setattr(context, key, payload)

        context.findings = list(context.analysis.get("findings") or [])

        if paper_path and os.path.exists(paper_path):
            context.paper_text = self._read_text(paper_path)
            context.available["paper"] = True
        else:
            context.available["paper"] = False
            context.missing.append("paper.*")

        return context

    @staticmethod
    def _resolve(path: str) -> Any:
        target = path or "."
        if os.path.isdir(target):
            directory = target
            paper_path = ""
            for name in ReviewContextLoader.PAPER_CANDIDATES:
                candidate = os.path.join(directory, name)
                if os.path.exists(candidate):
                    paper_path = candidate
                    break
            return directory, paper_path
        return os.path.dirname(os.path.abspath(target)), target

    @staticmethod
    def _read_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""


class AdversarialReviewer:
    """对论文逐条发起攻击；只找漏洞，不修复，也不加分。"""

    def __init__(
        self,
        min_primary_sample: int = DEFAULT_MIN_PRIMARY_SAMPLE,
        min_domains: int = DEFAULT_MIN_DOMAINS,
        min_repositories: int = DEFAULT_MIN_REPOSITORIES,
    ):
        self._min_primary_sample = max(1, min_primary_sample)
        self._min_domains = max(1, min_domains)
        self._min_repositories = max(1, min_repositories)

    def review(self, context: ReviewContext) -> AuditReport:
        attacks: List[Attack] = []
        checks: List[str] = []
        skipped: List[str] = []
        self_limited: List[str] = []

        attacks.extend(self._sample_size_attacks(context, checks, skipped))
        attacks.extend(self._weak_evidence_attacks(context, attacks, checks, skipped, self_limited))
        attacks.extend(self._control_group_attacks(context, checks, skipped))
        attacks.extend(self._causation_attacks(context, checks, skipped))
        attacks.extend(self._survivorship_attacks(context, checks, skipped))
        attacks.extend(self._multiple_comparison_attacks(context, checks, skipped))
        attacks.extend(self._overgeneralization_attacks(context, checks, skipped))

        numbered = [replace(attack, attack_id=f"A-{index:02d}") for index, attack in enumerate(attacks, 1)]
        return AuditReport(
            attacks=numbered,
            checks_run=checks,
            skipped=skipped,
            self_limited=sorted(set(self_limited)),
            context_available=dict(context.available),
            confidence=context.confidence,
        )

    # ------------------------------------------------------------------ 各类攻击

    def _claim_findings(self, context: ReviewContext) -> List[Dict[str, Any]]:
        """只攻击“主张了某种效应”的结论；已自我否定的结论不重复开火。"""

        claimed: List[Dict[str, Any]] = []
        for finding in context.findings:
            statement = str(finding.get("statement") or "")
            if any(marker in statement for marker in INSUFFICIENCY_MARKERS):
                continue
            metric = str(finding.get("metric") or "")
            if finding.get("significant") or metric in CLAIM_METRICS:
                claimed.append(finding)
        return claimed

    def _sample_size_attacks(self, context, checks, skipped) -> List[Attack]:
        if not context.findings:
            skipped.append("样本量攻击：没有 findings（需要 analysis.json）")
            return []

        checks.append(SAMPLE_SIZE)
        attacks: List[Attack] = []

        for finding in self._claim_findings(context):
            size = finding.get("sample_size")
            if not isinstance(size, int) or size >= self._min_primary_sample:
                continue

            if size <= 1:
                severity = FATAL
                statement = f"「{finding.get('statement')}」只有一个数据点，" "无法推广到任何总体，不能作为结论"
                fix = "移除该结论，改为“需要更多数据”"
            else:
                severity = MAJOR
                statement = (
                    f"「{finding.get('statement')}」的样本量 n={size}，"
                    f"低于把该结论当作主要发现的门槛 {self._min_primary_sample}"
                )
                fix = "把该结论降级为“初步观察”，不作为主要发现"

            attacks.append(
                Attack(
                    attack_id="",
                    attack_type=SAMPLE_SIZE,
                    severity=severity,
                    target=str(finding.get("finding_id") or ""),
                    finding_id=str(finding.get("finding_id") or ""),
                    category=str(finding.get("category") or "other"),
                    statement=statement,
                    evidence=f"sample_size={size}，门槛={self._min_primary_sample}",
                    suggested_fix=fix,
                )
            )
        return attacks

    def _weak_evidence_attacks(self, context, existing, checks, skipped, self_limited) -> List[Attack]:
        grades = context.grades
        if not grades:
            skipped.append("证据等级攻击：没有 grades（需要 integrity.json）")
            return []

        checks.append(WEAK_EVIDENCE)
        already_attacked = {attack.finding_id for attack in existing}
        attacks: List[Attack] = []

        for finding_id, grade in sorted(grades.items()):
            level = str(grade.get("grade") or "A")
            if level == "A" or finding_id in already_attacked:
                continue
            statement = str(grade.get("statement") or "")
            if any(marker in statement for marker in INSUFFICIENCY_MARKERS):
                # 结论已经自己声明“样本不足、不报告该指标”，再说它“以结果的口吻陈述”
                # 就是误伤；这里显式记录，而不是悄悄放过。
                self_limited.append(finding_id)
                continue

            severity = MAJOR if level == "C" else MINOR
            reasons = "；".join(str(item) for item in grade.get("reasons") or []) or "未说明"
            attacks.append(
                Attack(
                    attack_id="",
                    attack_type=WEAK_EVIDENCE,
                    severity=severity,
                    target=finding_id,
                    finding_id=finding_id,
                    category=str(grade.get("category") or "other"),
                    statement=(f"「{statement}」的证据等级只有 {level}，却仍以结果的口吻陈述"),
                    evidence=f"证据等级={level}（原因：{reasons}）",
                    suggested_fix="按证据等级重述或移除该结论",
                )
            )
        return attacks

    def _control_group_attacks(self, context, checks, skipped) -> List[Attack]:
        strategies = context.dataset.get("strategies") or []
        strategy_findings = [
            finding for finding in self._claim_findings(context) if str(finding.get("category")) == "strategy"
        ]
        if not strategy_findings:
            skipped.append("无对照组攻击：论文没有主张策略收益的结论，该检查不适用")
            return []

        checks.append(NO_CONTROL_GROUP)
        observations = len(strategies)
        detail = (
            f"共 {observations} 条策略观测，全部来自“策略已被应用”的事件"
            if observations
            else "策略收益取自被应用事件的自报收益"
        )
        if not strategies:
            skipped.append("无对照组攻击：dataset.json 里没有策略观测，只能用结论文本判断")

        return [
            Attack(
                attack_id="",
                attack_type=NO_CONTROL_GROUP,
                severity=MAJOR,
                target="策略收益类结论",
                category="strategy",
                statement=(
                    "策略收益类结论没有对照组：所有观测都发生在“策略已经被应用”之后，"
                    "无法排除同期其他变化带来的收益，因此不能把收益归因于策略本身"
                ),
                evidence=detail,
                suggested_fix="改为“与…一致/伴随正收益”，并把缺少对照组写进局限性",
            )
        ]

    def _causation_attacks(self, context, checks, skipped) -> List[Attack]:
        attacks: List[Attack] = []
        ran = False

        correlational = [
            finding for finding in context.findings if str(finding.get("metric") or "") in CORRELATIONAL_METRICS
        ]
        if correlational:
            ran = True
            for finding in correlational:
                statement = str(finding.get("statement") or "")
                marker = next((item for item in CAUSAL_MARKERS if item in statement), "")
                if not marker:
                    continue
                attacks.append(
                    Attack(
                        attack_id="",
                        attack_type=CORRELATION_CAUSATION,
                        severity=MAJOR,
                        target=str(finding.get("finding_id") or ""),
                        finding_id=str(finding.get("finding_id") or ""),
                        category=str(finding.get("category") or "other"),
                        statement=(f"「{statement}」用的是相关性证据，却使用了因果措辞“{marker}”"),
                        evidence=f"指标 {finding.get('metric')} 是相关性度量，不是干预实验",
                        suggested_fix="改为“与…一致/伴随”，不要使用因果动词",
                    )
                )

        if context.paper_text:
            ran = True
            attacks.extend(self._paper_causation_attacks(context))
        else:
            skipped.append("正文因果措辞检查：没有论文正文（需要 paper.md / paper.tex）")

        if ran:
            checks.append(CORRELATION_CAUSATION)
        return attacks

    def _paper_causation_attacks(self, context: ReviewContext) -> List[Attack]:
        attacks: List[Attack] = []
        for index, sentence in enumerate(SENTENCE_SPLIT_RE.split(context.paper_text)):
            text = sentence.strip()
            if not text or "证据等级" in text:
                continue
            if not any(term in text for term in CORRELATION_TERMS):
                continue
            marker = next((item for item in PAPER_CAUSAL_MARKERS if item in text), "")
            if not marker:
                continue
            attacks.append(
                Attack(
                    attack_id="",
                    attack_type=CORRELATION_CAUSATION,
                    severity=MAJOR,
                    target=f"正文第 {index + 1} 句",
                    statement=f"正文把相关性写成了因果：“{text[:80]}…”（使用了“{marker}”）",
                    evidence="正文同时出现相关性术语与因果动词",
                    suggested_fix="把因果动词改掉，或补上干预实验",
                )
            )
        return attacks

    def _survivorship_attacks(self, context, checks, skipped) -> List[Attack]:
        strategies = context.dataset.get("strategies")
        if not strategies:
            skipped.append("幸存者偏差检查：没有策略观测（需要 dataset.json）")
            return []

        checks.append(SURVIVORSHIP)
        benefits = [float(item.get("benefit", 0.0) or 0.0) for item in strategies]
        failures = sum(1 for value in benefits if value <= 0)
        if failures:
            return []

        return [
            Attack(
                attack_id="",
                attack_type=SURVIVORSHIP,
                severity=MAJOR,
                target="策略收益分布",
                category="strategy",
                statement=(
                    f"{len(benefits)} 条策略观测全部为正收益，0 条失败或无效记录："
                    "系统可能只记录了成功的尝试，据此得出的“策略有效”是幸存者偏差"
                ),
                evidence=f"positive={len(benefits)}, failures={failures}",
                suggested_fix="补充失败/回滚的记录，或明确声明失败样本未被采集",
            )
        ]

    def _multiple_comparison_attacks(self, context, checks, skipped) -> List[Attack]:
        analysis = context.analysis.get("analysis") or {}
        p_values: List[Any] = []
        tests = 0

        trend = analysis.get("success_rate_trend")
        if isinstance(trend, dict):
            tests += 1
            p_values.append(("趋势检验", trend.get("p_value")))

        correlation = analysis.get("similarity_correlation")
        if isinstance(correlation, dict):
            tests += 1
            p_values.append(("相似度-成功率相关", correlation.get("p_value")))

        comparisons = analysis.get("comparisons") or []
        tests += len(comparisons)
        p_values.extend((str(item.get("label")), item.get("p_value")) for item in comparisons)

        strategy_stats = analysis.get("strategy_stats") or []
        tests += len(strategy_stats)

        numeric = [
            (label, float(value))
            for label, value in p_values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not numeric:
            skipped.append("多重比较检查：analysis.json 里没有可用的 p 值")
            return []

        checks.append(MULTIPLE_COMPARISONS)
        alpha = context.alpha
        threshold = alpha / max(1, tests)
        best_label, best_p = min(numeric, key=lambda item: item[1])
        if best_p <= threshold:
            return []

        return [
            Attack(
                attack_id="",
                attack_type=MULTIPLE_COMPARISONS,
                severity=MAJOR,
                target=best_label,
                category="capability",
                statement=(
                    f"本文一共做了 {tests} 次假设检验，Bonferroni 校正后的显著性阈值是 "
                    f"{threshold:.5f}，而最小的 p 值是 {best_p:.5f}（{best_label}）——"
                    "按校正后标准没有任何结论显著，“显著”可能来自多重比较的假阳性"
                ),
                evidence=f"tests={tests}, alpha={alpha}, threshold={threshold:.6f}, min_p={best_p:.6f}",
                suggested_fix="报告校正后的 p 值，或把“显著”改为“未通过多重比较校正”",
            )
        ]

    def _overgeneralization_attacks(self, context, checks, skipped) -> List[Attack]:
        if not context.dataset:
            skipped.append("过度泛化检查：没有数据集（需要 dataset.json）")
            return []

        checks.append(OVERGENERALIZATION)
        attacks: List[Attack] = []

        pairs = context.dataset.get("similarity_pairs") or []
        domains = {
            str(item.get("domain") or "")
            for item in pairs
            if item.get("domain") and str(item.get("domain")) != "unknown"
        }
        if domains and len(domains) < self._min_domains:
            attacks.append(
                Attack(
                    attack_id="",
                    attack_type=OVERGENERALIZATION,
                    severity=MAJOR,
                    target="跨域迁移结论",
                    category="transfer",
                    statement=(
                        f"跨域迁移的全部证据只来自 {len(domains)} 个领域（{len(pairs)} 组配对），"
                        "不足以支撑关于跨域迁移的一般规律"
                    ),
                    evidence=f"domains={sorted(domains)}, pairs={len(pairs)}, 门槛={self._min_domains}",
                    suggested_fix="把结论限定在这些领域内，或补充更多领域的数据",
                )
            )

        detection = context.dataset.get("phase_detection") or {}
        commits = int(detection.get("commit_count") or 0)
        if commits:
            attacks.append(
                Attack(
                    attack_id="",
                    attack_type=OVERGENERALIZATION,
                    severity=MINOR,
                    target="演化阶段结论",
                    category="evolution",
                    statement=(
                        f"演化阶段结论来自单一仓库的 {commits} 次提交（{len(context.dataset.get('phases') or [])} 个阶段），"
                        f"少于一般规律所需的 {self._min_repositories} 个独立仓库"
                    ),
                    evidence=f"repositories=1, commits={commits}",
                    suggested_fix="把结论限定为“本仓库的演化过程”，不要外推为普遍规律",
                )
            )
        return attacks
