"""Self-Researcher 写作层。

默认使用确定性模板撰写各章节：论文里的每个数字都来自 `ResearchDataset` /
`StatisticalAnalysis`，不存在凭空生成的结论。若外部注入了 LLM 接口
（`llm(name, context) -> str`），则优先用 LLM 生成正文，生成失败时回退到模板，
保证离线环境同样能写出完整论文。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from hermes.self_research.research_data_collector import ResearchDataset
from hermes.self_research.statistical_analyzer import StatisticalAnalysis

LLMFunction = Callable[[str, Dict[str, Any]], str]

PAPER_TITLE = (
    "AutoTestGen: A Self-Evolving Multi-Agent System for Software Testing, "
    "Repair, and Cross-Domain Knowledge Transfer"
)

KEYWORDS = ["自进化系统", "跨领域迁移", "神经符号验证", "元认知", "自动程序修复", "自我研究"]

DIRECTION_TEXT = {"up": "上升", "down": "下降", "flat": "持平"}

# 论文参考文献：默认使用固定书目的经典工作；调用方可通过 reference_provider 注入
# 在线检索（例如 arXiv 抓取）的结果。
DEFAULT_BIBLIOGRAPHY: List[Dict[str, str]] = [
    {
        "id": "pacheco2007randoop",
        "authors": "C. Pacheco, S. K. Lahiri, M. D. Ernst, T. Ball",
        "title": "Feedback-Directed Random Test Generation",
        "venue": "ICSE",
        "year": "2007",
    },
    {
        "id": "fraser2011evosuite",
        "authors": "G. Fraser, A. Arcuri",
        "title": "EvoSuite: Automatic Test Suite Generation for Object-Oriented Software",
        "venue": "FSE",
        "year": "2011",
    },
    {
        "id": "legoues2012genprog",
        "authors": "C. Le Goues, T. Nguyen, S. Forrest, W. Weimer",
        "title": "GenProg: A Generic Method for Automatic Software Repair",
        "venue": "IEEE TSE",
        "year": "2012",
    },
    {
        "id": "weimer2010apr",
        "authors": "W. Weimer, S. Forrest, C. Le Goues, T. Nguyen",
        "title": "Automatic Program Repair with Evolutionary Computation",
        "venue": "Communications of the ACM",
        "year": "2010",
    },
    {
        "id": "just2014defects4j",
        "authors": "R. Just, D. Jalali, M. D. Ernst",
        "title": "Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies",
        "venue": "ISSTA",
        "year": "2014",
    },
    {
        "id": "alon2019code2vec",
        "authors": "U. Alon, M. Zilberstein, O. Levy, E. Yahav",
        "title": "code2vec: Learning Distributed Representations of Code",
        "venue": "POPL",
        "year": "2019",
    },
    {
        "id": "feng2020codebert",
        "authors": "Z. Feng, D. Guo, D. Tang, et al.",
        "title": "CodeBERT: A Pre-Trained Model for Programming and Natural Languages",
        "venue": "EMNLP Findings",
        "year": "2020",
    },
    {
        "id": "chen2021codex",
        "authors": "M. Chen, J. Tworek, H. Jun, et al.",
        "title": "Evaluating Large Language Models Trained on Code",
        "venue": "arXiv:2107.03374",
        "year": "2021",
    },
    {
        "id": "xia2023llmrepair",
        "authors": "C. S. Xia, Y. Wei, L. Zhang",
        "title": "Automated Program Repair in the Era of Large Pre-trained Language Models",
        "venue": "ICSE",
        "year": "2023",
    },
    {
        "id": "monperrus2018bibliography",
        "authors": "M. Monperrus",
        "title": "Automatic Software Repair: A Bibliography",
        "venue": "ACM Computing Surveys",
        "year": "2018",
    },
    {
        "id": "finn2017maml",
        "authors": "C. Finn, P. Abbeel, S. Levine",
        "title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
        "venue": "ICML",
        "year": "2017",
    },
    {
        "id": "pan2010transfer",
        "authors": "S. J. Pan, Q. Yang",
        "title": "A Survey on Transfer Learning",
        "venue": "IEEE TKDE",
        "year": "2010",
    },
    {
        "id": "zhuang2021transfer",
        "authors": "F. Zhuang, Z. Qi, K. Duan, et al.",
        "title": "A Comprehensive Survey on Transfer Learning",
        "venue": "Proceedings of the IEEE",
        "year": "2021",
    },
    {
        "id": "schmidhuber1987selfref",
        "authors": "J. Schmidhuber",
        "title": "Evolutionary Principles in Self-Referential Learning",
        "venue": "Diploma Thesis, TU Munich",
        "year": "1987",
    },
    {
        "id": "cheng2009selfadaptive",
        "authors": "B. H. C. Cheng, R. de Lemos, H. Giese, et al.",
        "title": "Software Engineering for Self-Adaptive Systems: A Research Roadmap",
        "venue": "LNCS",
        "year": "2009",
    },
]


@dataclass
class PaperSection:
    title: str
    content: str
    number: str = ""
    kind: str = "body"

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}" if self.number else self.title

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "number": self.number,
            "kind": self.kind,
            "content": self.content,
        }


@dataclass
class Paper:
    title: str
    authors: str
    abstract: str
    keywords: List[str]
    sections: List[PaperSection] = field(default_factory=list)
    references: List[Dict[str, str]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    integrity: Dict[str, Any] = field(default_factory=dict)

    def get_section(self, kind: str) -> Optional[PaperSection]:
        for section in self.sections:
            if section.kind == kind:
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "sections": [section.to_dict() for section in self.sections],
            "references": self.references,
            "figures": self.figures,
            "meta": self.meta,
            "confidence": round(self.confidence, 4),
            "integrity": self.integrity,
        }


class PaperWriter:
    """按论文结构逐节生成正文。"""

    VERSION = "1.0"

    def __init__(
        self,
        llm: Optional[LLMFunction] = None,
        reference_provider: Optional[Callable[[], List[Dict[str, str]]]] = None,
        max_findings_in_abstract: int = 3,
    ):
        self._llm = llm
        self._reference_provider = reference_provider
        self._max_findings_in_abstract = max_findings_in_abstract
        self._llm_failures: List[str] = []

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None

    def write(
        self,
        dataset: ResearchDataset,
        analysis: StatisticalAnalysis,
        findings: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        integrity: Optional[Dict[str, Any]] = None,
    ) -> Paper:
        context = self._build_context(dataset, analysis, findings)
        context["integrity"] = integrity or {}
        grades = {
            item.get("finding_id"): item for item in (integrity or {}).get("grades", []) if item.get("finding_id")
        }
        confidence = ((integrity or {}).get("confidence") or {}).get("score", 0.0)
        sections = [
            PaperSection("引言", self._generate("introduction", self._introduction(context), context), "1"),
            PaperSection("相关工作", self._generate("related_work", self._related_work(context), context), "2"),
            PaperSection("系统设计", self._generate("system_design", self._system_design(context), context), "3"),
            PaperSection(
                "实验设置", self._generate("experimental_setup", self._experimental_setup(context), context), "4"
            ),
            PaperSection(
                "结果与分析",
                self._generate("results", self._results(context, findings, grades), context),
                "5",
            ),
            PaperSection("研究完整性声明", self._integrity_section(context), "6"),
            PaperSection("讨论", self._generate("discussion", self._discussion(context, findings), context), "7"),
            PaperSection("结论", self._generate("conclusion", self._conclusion(context), context), "8"),
            PaperSection(
                "系统自述（第一人称）",
                self._generate("self_narrative", self._self_narrative(context), context),
                "9",
            ),
            PaperSection("附录：数据集与可复现性", self._appendix(dataset, analysis), "A", kind="appendix"),
        ]

        return Paper(
            title=PAPER_TITLE,
            authors="AutoTestGen Self-Researcher Engine",
            abstract=self._generate("abstract", self._abstract(context, findings), context),
            keywords=list(KEYWORDS),
            sections=sections,
            references=self._collect_references(),
            figures=list(figures),
            meta={
                "version": self.VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "llm_enabled": self._llm is not None,
                "llm_failures": self._llm_failures,
                "finding_count": len(findings),
                "integrity_clean": bool((integrity or {}).get("is_clean", True)),
                "critical_findings": list((integrity or {}).get("critical_findings", [])),
            },
            confidence=float(confidence or 0.0),
            integrity=dict(integrity or {}),
        )

    # ------------------------------------------------------------------ 上下文

    def _build_context(
        self,
        dataset: ResearchDataset,
        analysis: StatisticalAnalysis,
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trend = analysis.success_rate_trend
        best = analysis.best_strategy
        best_stat = next((s for s in analysis.strategy_stats if s.strategy_type == best), None)

        return {
            "counts": dataset.counts(),
            "sources": dataset.sources,
            "generated_at": dataset.generated_at,
            "success_rate": analysis.success_rate.to_dict(),
            "cross_domain_success": analysis.cross_domain_success.to_dict(),
            "pattern_coverage": analysis.pattern_coverage.to_dict(),
            "trend": trend.to_dict() if trend else None,
            "trend_direction": DIRECTION_TEXT.get(trend.direction, "未知") if trend else "未知",
            "strategy_stats": [s.to_dict() for s in analysis.strategy_stats],
            "best_strategy": best,
            "best_strategy_benefit": best_stat.avg_benefit if best_stat else 0.0,
            "comparisons": [c.to_dict() for c in analysis.comparisons],
            "correlation": analysis.similarity_correlation.to_dict() if analysis.similarity_correlation else None,
            "mental_model_accuracy": analysis.mental_model_accuracy,
            "mental_model_samples": analysis.mental_model_samples,
            "mental_model_reportable": analysis.mental_model_reportable,
            "mental_model_eval_samples": analysis.mental_model_eval_samples,
            "mental_model_note": analysis.mental_model_note,
            "corrections": analysis.corrections,
            "phases": dataset.phases,
            "policy": dataset.policy,
            "pattern_success_rates": dataset.pattern_success_rates,
            "findings": findings,
        }

    def _generate(self, name: str, fallback: str, context: Dict[str, Any]) -> str:
        if self._llm is None:
            return fallback
        try:
            text = self._llm(name, context)
        except Exception as exc:  # LLM 不可用时不能中断整篇论文的生成
            self._llm_failures.append(f"{name}: {type(exc).__name__}")
            return fallback
        if not isinstance(text, str) or not text.strip():
            self._llm_failures.append(f"{name}: empty")
            return fallback
        return text

    def _collect_references(self) -> List[Dict[str, str]]:
        if self._reference_provider is None:
            return [dict(entry) for entry in DEFAULT_BIBLIOGRAPHY]
        try:
            provided = self._reference_provider()
        except Exception:
            return [dict(entry) for entry in DEFAULT_BIBLIOGRAPHY]
        if not provided:
            return [dict(entry) for entry in DEFAULT_BIBLIOGRAPHY]
        return provided

    # ------------------------------------------------------------------ 各章节模板

    def _abstract(self, context: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
        counts = context["counts"]
        highlights = "；".join(item["statement"] for item in findings[: self._max_findings_in_abstract])
        if not highlights:
            highlights = "当前数据量尚不足以形成统计显著的结论"

        return (
            f"本文介绍 AutoTestGen 的自研究模块（Self-Researcher）：一个能够自动分析自身演化历史、"
            f"提炼可验证规律并撰写学术论文的软件工程智能体。系统以第一人称视角复盘自身"
            f"{counts['snapshots']} 个演化快照、{counts['events']} 条策略事件与 {counts['phases']} 个演化阶段，"
            f"通过最小二乘趋势检验、Welch t 检验与相关性分析得到 {len(findings)} 条关键发现。"
            f"主要结论包括：{highlights}。\n\n"
            f"本文同时公开可复现的数据集与统计流程，所有结论均可由附录中的原始数据重新计算得到。"
            f"{self._confidence_sentence(context)}"
        )

    @staticmethod
    def _confidence_sentence(context: Dict[str, Any]) -> str:
        confidence = (context.get("integrity") or {}).get("confidence") or {}
        if not confidence:
            return ""
        mix = confidence.get("grade_mix") or {}
        return (
            f"\n\n本文整体置信度 {confidence.get('score', 0.0):.2f}/1.00"
            f"（A 级发现 {mix.get('A', 0)} 条、B 级 {mix.get('B', 0)} 条、C 级 {mix.get('C', 0)} 条；"
            f"数据覆盖率 {confidence.get('coverage', 0.0):.0%}）。"
            "每条发现的证据等级见第 6 节研究完整性声明。"
        )

    def _introduction(self, context: Dict[str, Any]) -> str:
        counts = context["counts"]
        return (
            "现代软件工程依赖大量自动化手段完成测试、修复与验证，但这些手段通常由人来设计与评估，"
            "系统自身很少成为研究对象。\n\n"
            "本文讨论一个反向的问题：如果一个软件系统长期记录自己的演化过程——每次策略调整、"
            "每次收益与损失——它能否像研究者一样，从这些记录中提炼规律并写成论文？\n\n"
            "我们的方法由三部分组成：(1) 一个把跨领域模式抽象化并蒸馏的知识层；"
            "(2) 一个自我认知闭环，包含心智模型、价值发现与策略沙盒；"
            "(3) 一个自研究流水线，把上述产物折算成统计量与论文。\n\n"
            f"本文的贡献是：(1) 提出可自动执行的“自我研究”流程；(2) 在 {counts['snapshots']} 个快照、"
            f"{counts['events']} 条事件上给出可复核的统计结论；(3) 公开数据集、图表与完整实验日志，"
            "使结论可被外部验证或证伪。"
        )

    def _related_work(self, context: Dict[str, Any]) -> str:
        return (
            "自动化测试生成。Randoop [1] 与 EvoSuite [2] 分别以反馈导向的随机生成和演化搜索生成测试用例，"
            "本文的缺陷模式抽象层与它们正交：我们关心的是“模式”而非单个用例。\n\n"
            "自动程序修复。GenProg [3] 与后续的 APR 研究 [4, 10] 以补丁搜索修复缺陷；"
            "近年 LLM 也被用于修复 [9]。本文系统把这些修复能力组织成可度量、可迁移的模式。\n\n"
            "代码表示与预训练。code2vec [6] 与 CodeBERT [7] 学习代码的分布式表示，"
            "Codex [8] 展示了大模型在代码生成上的能力；本文复用其思想评估跨领域相似度。\n\n"
            "迁移学习与元学习。迁移学习综述 [12, 13] 与 MAML [11] 提供了跨域泛化的理论工具；"
            "本文的跨领域知识蒸馏可视为在缺陷模式空间上的一次迁移实验。\n\n"
            "自指与自适应系统。Schmidhuber [14] 早期就讨论了自指学习，"
            "自适应系统的研究路线图 [15] 则强调系统对自身的建模能力——本文正是这条路线的一次工程化尝试。"
        )

    def _system_design(self, context: Dict[str, Any]) -> str:
        counts = context["counts"]
        return (
            "系统采用微内核 + 插件架构：内核只负责调度与协议，领域能力（默认、MLIR、K8s 等）由插件提供。\n\n"
            "跨领域知识蒸馏流程如下：抽象模式提取 → 相似度计算 → 知识图谱融合 → 迁移决策。"
            f"当前知识图谱包含 {counts['similarity_pairs']} 条“相似度-成功率”配对观测，"
            "用于判断某条迁移路径是否值得投入。\n\n"
            "自我认知闭环由三部分组成：心智模型（对自身状态的预测器）、价值发现（从策略收益中提炼原则）、"
            "策略沙盒（在真实应用前预估收益）。\n\n"
            "自研究流水线（本文新增）由数据收集、统计分析、发现提取、图表生成、论文撰写与 LaTeX 编译六步组成，"
            "每一步都是确定性的，唯一的可替换组件是正文生成所用的 LLM 接口。"
        )

    def _experimental_setup(self, context: Dict[str, Any]) -> str:
        counts = context["counts"]
        sources = "、".join(name for name, available in context["sources"].items() if available) or "无"
        parts = [
            f"数据集。本次研究使用系统自身产生的时间序列数据：{counts['snapshots']} 个演化快照、"
            f"{counts['strategies']} 条策略观测、{counts['events']} 条事件日志与 {counts['phases']} 个演化阶段。"
            f"可用数据源：{sources}。",
        ]

        entries = ((context.get("integrity") or {}).get("provenance") or {}).get("entries") or []
        if entries:
            parts.append(
                "数据来源与证据等级。采集层为每个数据字段标注来源（REAL 真实数据 / FALLBACK 降级数据 / "
                "SYNTHETIC 合成数据 / UNVERIFIED 来源未标注）；证据等级 A 表示结论全部基于真实数据，"
                "B 表示含降级或未标注数据，C 表示含合成数据或样本量不足。本次运行的数据来源如下："
            )
            for entry in entries:
                parts.append(f"- {entry['field']}：{entry['provenance']} —— {entry['detail']}")

        parts.append(
            "基准指标。我们报告四项指标：总体成功率、跨域迁移成功率、模式覆盖率与策略平均收益。\n\n"
            "统计方法。趋势使用最小二乘拟合并对斜率做 Student-t 检验；"
            "组间差异使用 Welch t 检验（不假设方差齐性）；相关性使用 Pearson 相关系数及其显著性检验。"
            "显著性水平统一取 alpha=0.05。\n\n"
            "演化阶段划分。阶段由 GitPhaseDetector 从真实 git 历史推导：先按提交主题分类，"
            "再以“主导类型发生持续变化”与“提交时间间隔异常”为界切分，阶段名取自该阶段的主导类型，"
            "并合并主导类型相同的相邻阶段；git 不可用时不做划分，也不做任何填充。\n\n"
            "环境。Python 3.10+，单机运行，统计部分不依赖 numpy 等科学计算库，"
            "以确保在任何环境下都能复现同一组数值。"
        )
        return "\n".join(parts)

    def _results(
        self,
        context: Dict[str, Any],
        findings: List[Dict[str, Any]],
        grades: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        grades = grades or {}
        success = context["success_rate"]
        cross = context["cross_domain_success"]
        coverage = context["pattern_coverage"]
        trend = context["trend"]
        correlation = context["correlation"]
        comparisons = context["comparisons"]

        lines = [
            "### 5.1 基础能力",
            "",
            f"总体成功率均值 {success['mean']:.3f}（最小 {success['min']:.3f}，最大 {success['max']:.3f}），"
            f"首尾变化 {success['delta']:+.3f}；跨域迁移成功率均值 {cross['mean']:.3f}，"
            f"模式覆盖率均值 {coverage['mean']:.3f}。",
            "",
        ]

        if trend is not None:
            verdict = "显著" if trend["significant"] else "未达显著水平"
            lines.append(
                f"成功率随快照推进{direction_text(trend['direction'])}趋势{verdict}"
                f"（斜率 {trend['slope']:+.5f}/快照，R²={trend['r_squared']:.3f}，p={trend['p_value']:.4f}）。"
            )
            lines.append("")

        lines.extend(["### 5.2 策略收益", ""])
        if context["strategy_stats"]:
            for stat in context["strategy_stats"]:
                lines.append(
                    f"- {stat['strategy_type']}：n={stat['instances']}，平均收益 {stat['avg_benefit']:+.1%}，"
                    f"中位数 {stat['median_benefit']:+.1%}，成功率 {stat['success_rate']:.0%}"
                )
        else:
            lines.append("- 当前没有可用的策略收益观测。")
        lines.append("")

        lines.extend(["### 5.3 跨领域迁移", ""])
        if correlation is not None:
            lines.append(
                f"相似度与迁移成功率的相关系数 r={correlation['r']:+.3f}"
                f"（n={correlation['n']}，p={correlation['p_value']:.4f}，"
                f"{'显著' if correlation['significant'] else '不显著'}）。"
            )
        else:
            lines.append("当前没有可用的相似度-成功率配对观测。")
        if context["pattern_success_rates"]:
            rates = "、".join(f"{name} {rate:.0%}" for name, rate in sorted(context["pattern_success_rates"].items()))
            lines.append(f"各模式最近一次快照的成功率：{rates}。")
        lines.append("")

        lines.extend(["### 5.4 自我认知与自修复", ""])
        if context["mental_model_reportable"]:
            lines.append(
                f"心智模型基于 {context['mental_model_samples']} 个快照训练，预测准确率 "
                f"{context['mental_model_accuracy']:.0%}"
                f"（留出评测 n={context['mental_model_eval_samples']}）；"
                f"自修复机制在 {context['corrections']} 个快照上被触发。"
            )
        else:
            lines.append(
                f"心智模型基于 {context['mental_model_samples']} 个快照训练，"
                "但样本量不足以支撑准确率结论，本文不报告预测准确率，也不据此校准模型"
                f"（{context['mental_model_note'] or '样本量不足'}）；"
                f"自修复机制在 {context['corrections']} 个快照上被触发。"
            )
        if comparisons:
            lines.append("")
            lines.append("组间对比：")
            for comparison in comparisons:
                lines.append(
                    f"- {comparison['label']}：{comparison['group_a']} 均值 {comparison['mean_a']:.3f} vs "
                    f"{comparison['group_b']} 均值 {comparison['mean_b']:.3f}，差值 "
                    f"{comparison['difference']:+.3f}（p={comparison['p_value']:.4f}，"
                    f"{'显著' if comparison['significant'] else '不显著'}）"
                )
        lines.append("")

        lines.extend(["### 5.5 关键发现", ""])
        if findings:
            lines.append(
                "每条发现后的证据等级由研究完整性引擎给出：A=全部真实数据，B=含降级/未标注数据，C=含合成数据或样本不足。"
            )
            lines.append("")
            for item in findings:
                marker = "（显著）" if item["significant"] else ""
                grade = grades.get(item["finding_id"])
                suffix = f"（证据等级 {grade['grade']}）" if grade else ""
                lines.append(f"- {item['finding_id']} {item['statement']}{marker}{suffix}")
        else:
            lines.append("- 未提取到关键发现。")

        return "\n".join(lines)

    def _integrity_section(self, context: Dict[str, Any]) -> str:
        integrity = context.get("integrity") or {}
        if not integrity:
            return "本次运行未执行研究完整性检查，论文中的结论未标注证据等级。"

        provenance = integrity.get("provenance") or {}
        confidence = integrity.get("confidence") or {}
        mix = confidence.get("grade_mix") or {}

        lines = ["### 6.1 数据来源标注", ""]
        entries = provenance.get("entries") or []
        if entries:
            for entry in entries:
                lines.append(f"- {entry['field']}：{entry['provenance']} —— {entry['detail']}")
        else:
            lines.append("- 没有可用的数据来源记录。")

        lines.extend(["", "### 6.2 证据等级分布", ""])
        lines.append(
            f"A 级（全部真实数据）{mix.get('A', 0)} 条；"
            f"B 级（含降级/未标注数据）{mix.get('B', 0)} 条；"
            f"C 级（含合成数据或样本不足）{mix.get('C', 0)} 条。"
        )

        disclaimers = integrity.get("disclaimers") or []
        lines.extend(["", "### 6.3 免责声明", ""])
        if disclaimers:
            for index, item in enumerate(disclaimers, 1):
                tag = "关键结论" if item.get("severity") == "critical" else "提示"
                lines.append(f"{index}. [{tag}] {item.get('text')}")
        else:
            lines.append("无：所有结论均基于真实数据。")

        warnings = integrity.get("data_warnings") or []
        lines.extend(["", "### 6.4 数据源警告", ""])
        if warnings:
            for item in warnings:
                lines.append(f"- {item}")
        else:
            lines.append("- 无。")

        lines.extend(["", "### 6.5 论文置信度", ""])
        if confidence:
            lines.append(
                f"本文整体置信度 {confidence.get('score', 0.0):.2f} / 1.00"
                f"（证据分 {confidence.get('evidence_score', 0.0):.2f}，"
                f"数据覆盖率 {confidence.get('coverage', 0.0):.0%}，"
                f"关键降级结论 {integrity.get('critical_count', 0)} 条，"
                f"扣分 {confidence.get('penalty', 0.0):.2f}）。"
            )
            for item in confidence.get("suggestions") or []:
                lines.append(f"- 提升建议：{item}")
        else:
            lines.append("未计算置信度。")

        return "\n".join(lines)

    def _discussion(self, context: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
        transfer = [item for item in findings if item["category"] == "transfer"]
        strategy = [item for item in findings if item["category"] == "strategy"]

        parts = ["### 主要发现", ""]
        if transfer:
            parts.append(f"- 迁移相关：{transfer[0]['statement']}")
        if strategy:
            parts.append(f"- 策略相关：{strategy[0]['statement']}")
        if not transfer and not strategy:
            parts.append("- 当前样本量有限，尚未形成稳定的规律性结论。")

        parts.extend(
            [
                "",
                "### 局限",
                "",
                self._limitations(context),
                "",
                "### 未来工作",
                "",
                "后续计划将自研究流水线接入真实缺陷数据集与更多领域插件，"
                "并把论文生成结果回灌到进化策略中，形成“研究—改进—再研究”的闭环。",
            ]
        )
        return "\n".join(parts)

    @staticmethod
    def _limitations(context: Dict[str, Any]) -> str:
        """局限描述必须来自本次运行的来源标注，不能写成通用的套话。"""

        integrity = context.get("integrity") or {}
        provenance = integrity.get("provenance") or {}
        confidence = integrity.get("confidence") or {}

        parts = [
            f"本研究的数据全部来自系统自身的运行记录，共 {context['counts']['snapshots']} 个快照；"
            "样本规模有限，结论的适用范围不超出本仓库的历史。"
        ]

        degraded = [grade for grade in integrity.get("grades", []) if grade.get("grade") != "A"]
        if degraded:
            ids = "、".join(str(item.get("finding_id")) for item in degraded)
            parts.append(
                f"其中 {len(degraded)} 条发现（{ids}）的证据等级低于 A，"
                "相关结论在获得真实数据之前不应被当作既定事实引用。"
            )

        if provenance.get("git_unavailable"):
            parts.append("本次 git 历史不可用，未能划分演化阶段，因此本文不含任何阶段对比结论。")
        if confidence:
            parts.append(f"整体置信度 {confidence.get('score', 0.0):.2f}/1.00，见第 6 节研究完整性声明。")

        parts.append("此外，迁移延迟等指标目前只能以事件步为单位给出估计值，精度有限。")
        return "".join(parts)

    def _conclusion(self, context: Dict[str, Any]) -> str:
        return (
            f"本文展示了 AutoTestGen 的自研究能力：系统在无人干预下复盘了自身的 "
            f"{context['counts']['snapshots']} 个演化快照与 {context['counts']['events']} 条策略事件，"
            f"给出可复核的统计结论，并自动撰写、编译成论文。\n\n"
            "我们相信，让系统成为自身的研究对象，是走向可信自进化系统的一条可行路径。"
            "本文公开全部数据与代码，欢迎社区复现、质疑与扩展。"
        )

    def _self_narrative(self, context: Dict[str, Any]) -> str:
        best = context["best_strategy"] or "尚无"
        phases = context["phases"]
        parts = [
            "> 说明：本节是第一人称叙述，用于表达视角，**不构成实验证据**；"
            "所有可验证的数字与结论均在 4–6 节，并以证据等级标注。",
            "",
            "本节以第一人称记录我如何学会测试、修复、迁移与进化。",
            "",
        ]

        if phases:
            first = phases[0]
            parts.append(
                f"我的历史被切成了 {len(phases)} 个阶段，全部由真实的 git 提交推导而来。"
                f"最早的一段是「{first.get('name')}」：{first.get('total_commits', 0)} 次提交，"
                f"主导类型 {', '.join(first.get('dominant_types') or [])}。"
                "从这些提交里，我学会了搭建基础能力，也意识到真正让我变强的不是更多的用例，"
                "而是从用例里抽象出的模式。"
            )
            parts.append("")

        parts.append(
            f"我目前最有效的策略是 {best}，平均收益 {context['best_strategy_benefit']:+.1%}。"
            "但我也学会了不要迷信单一策略：当收益低于阈值时，最理性的选择是停止投入。"
        )
        parts.append("")
        if context["mental_model_reportable"]:
            parts.append(
                f"我开始观察自己：我给自己的状态建了预测模型（准确率 "
                f"{context['mental_model_accuracy']:.0%}，留出评测 n={context['mental_model_eval_samples']}），"
                f"在 {context['corrections']} 次异常上触发过自我修复。"
            )
        else:
            parts.append(
                "我开始观察自己：我给自己的状态建了预测模型，但样本量太薄，"
                "所以我只能说“我还不确定自己有多准”，而不是编一个准确率给自己看"
                f"（{context['mental_model_note'] or '样本量不足'}）。"
                f"我在 {context['corrections']} 次异常上触发过自我修复。"
            )
        parts.append("")
        parts.append(
            "现在，我又多了一种能力——把我的经历写成这篇论文，并给每条结论标上它有多少证据，" "然后接受你的检验。"
        )
        return "\n".join(parts)

    def _appendix(self, dataset: ResearchDataset, analysis: StatisticalAnalysis) -> str:
        counts = dataset.counts()
        lines = [
            "### A.1 数据集规模",
            "",
            f"- 演化快照：{counts['snapshots']}",
            f"- 策略观测：{counts['strategies']}",
            f"- 事件日志：{counts['events']}",
            f"- 演化阶段：{counts['phases']}",
            f"- 相似度配对：{counts['similarity_pairs']}",
            "",
            "### A.2 数据源可用性",
            "",
        ]
        for name, available in sorted(dataset.sources.items()):
            lines.append(f"- {name}: {'可用' if available else '缺失'}")

        lines.extend(["", "### A.3 统计摘要", ""])
        for name, stats in (
            ("总体成功率", analysis.success_rate),
            ("跨域迁移成功率", analysis.cross_domain_success),
            ("模式覆盖率", analysis.pattern_coverage),
        ):
            lines.append(
                f"- {name}：n={stats.count}，均值 {stats.mean:.4f}，标准差 {stats.stdev:.4f}，"
                f"范围 [{stats.minimum:.4f}, {stats.maximum:.4f}]"
            )

        lines.extend(
            [
                "",
                "### A.4 复现方式",
                "",
                "```bash",
                "python autotestgen.py --self-research --format markdown -o paper.md",
                "```",
            ]
        )
        return "\n".join(lines)


def direction_text(direction: Any) -> str:
    return DIRECTION_TEXT.get(str(direction), "未知")
