"""对抗性审计（Adversarial Review）的单元测试与端到端测试。

覆盖范围：审计上下文的加载与“缺产物就如实跳过”、7 类攻击规则、意见排序、
回应硬规则（只有次要意见可被反驳）、论文修订的单调性（只能变弱），以及
`--adversarial-review` CLI 入口的端到端行为。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import uuid

import pytest

from hermes.self_research import (
    AdversarialReviewer,
    Attack,
    PaperRevisionEngine,
    RebuttalGenerator,
    ReviewContext,
    ReviewContextLoader,
    WeaknessRanker,
)
from hermes.self_research.adversarial_reviewer import (
    CORRELATION_CAUSATION,
    FATAL,
    MAJOR,
    MINOR,
    MULTIPLE_COMPARISONS,
    NO_CONTROL_GROUP,
    OVERGENERALIZATION,
    SAMPLE_SIZE,
    SURVIVORSHIP,
    WEAK_EVIDENCE,
)
from hermes.self_research.rebuttal_generator import (
    ACCEPT,
    PARTIAL,
    REBUT,
    REVISION_DOWNGRADE,
    REVISION_REWORD,
    REVISION_WITHDRAW,
)
from hermes.self_research.weakness_ranker import (
    ACTION_REWORD,
    ACTION_WITHDRAW,
    RankedWeakness,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def work_dir():
    """临时工作目录（不用 tmp_path：受限环境下系统临时目录不可写）。"""

    base = os.path.join(REPO_ROOT, ".tmp_test", "adversarial_review")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _import_autotestgen():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import autotestgen

    return autotestgen


# ------------------------------------------------------------------ 构造工具


def _finding(finding_id, statement, metric="avg_benefit", category="strategy", sample_size=7, significant=True):
    return {
        "finding_id": finding_id,
        "statement": statement,
        "metric": metric,
        "category": category,
        "sample_size": sample_size,
        "significant": significant,
    }


def _grade(finding_id, statement, grade="A", category="strategy", reasons=()):
    return {
        "finding_id": finding_id,
        "statement": statement,
        "grade": grade,
        "category": category,
        "reasons": list(reasons),
    }


def _rich_context(**overrides) -> ReviewContext:
    """一份“所有产物都在”的上下文；用例按需覆盖单个字段。"""

    findings = [
        _finding("F-02", "策略 apply_proposal 显著有效：平均收益 +46.9%", sample_size=7),
        _finding("F-05", "策略 reset_similarity 显著有效：平均收益 +5.1%", sample_size=2),
        _finding(
            "F-06",
            "相似度越高，跨域迁移成功率越高",
            metric="transfer_strength",
            category="transfer",
            sample_size=4,
        ),
        _finding(
            "F-07",
            "自身仓库的演化可划分为 4 个阶段",
            metric="evolution_phases",
            category="evolution",
            sample_size=53,
            significant=False,
        ),
        _finding(
            "F-09",
            "跨域迁移中位间隔约 80 个事件步",
            metric="transfer_latency",
            category="transfer",
            sample_size=40,
        ),
        _finding(
            "F-10",
            "心智模型预测准确率 0.65",
            metric="prediction_accuracy",
            category="metacognition",
            sample_size=30,
        ),
    ]
    grades = [
        _grade("F-02", findings[0]["statement"]),
        _grade("F-05", findings[1]["statement"], grade="C", reasons=["样本量不足：n=2 < 3"]),
        _grade("F-06", findings[2]["statement"], grade="B", category="transfer", reasons=["估计值"]),
        _grade("F-07", findings[3]["statement"], category="evolution"),
        _grade("F-09", findings[4]["statement"], grade="C", category="transfer", reasons=["估计值"]),
        _grade("F-10", findings[5]["statement"], grade="B", category="metacognition", reasons=["留出样本不足"]),
    ]
    values = {
        "directory": "",
        "paper_path": "",
        "paper_text": "",
        "dataset": {
            "strategies": [{"benefit": 0.1}, {"benefit": 0.2}],
            "similarity_pairs": [{"domain": "a", "similarity": 0.8, "success": 0.7}],
            "phase_detection": {"commit_count": 53},
            "phases": [{}, {}, {}, {}],
        },
        "analysis": {
            "analysis": {"success_rate_trend": {"p_value": 0.001}, "similarity_correlation": {"p_value": 0.2}},
            "findings": findings,
        },
        "findings": findings,
        "integrity": {
            "grades": grades,
            "confidence": {"score": 0.9},
            "provenance": {"entries": [], "missing_sources": [], "git_unavailable": False, "notes": []},
        },
        "available": {"dataset": True, "analysis": True, "integrity": True, "paper": True},
        "missing": [],
    }
    values.update(overrides)
    return ReviewContext(**values)


def _attack(attack_type, severity, finding_id="", category="other", evidence="evidence"):
    return Attack(
        attack_id="A-01",
        attack_type=attack_type,
        severity=severity,
        target=finding_id or "全局结论",
        statement="审稿意见",
        evidence=evidence,
        suggested_fix="按意见修改",
        finding_id=finding_id,
        category=category,
    )


# ------------------------------------------------------------------ 上下文加载


class TestReviewContextLoader:
    def test_missing_artifacts_are_reported_not_invented(self, work_dir):
        (work_dir / "paper.md").write_text("# 只有正文\n", encoding="utf-8")

        context = ReviewContextLoader().load(str(work_dir))

        assert context.available["paper"] is True
        assert context.available["dataset"] is False
        assert set(context.missing) == {"dataset.json", "analysis.json", "integrity.json"}
        assert context.dataset == {}
        assert context.findings == []
        assert context.grades == {}
        assert context.confidence is None

    def test_loads_every_artifact_and_the_directory_paper(self, work_dir):
        (work_dir / "dataset.json").write_text(json.dumps({"strategies": []}), encoding="utf-8")
        (work_dir / "analysis.json").write_text(json.dumps({"findings": [{"finding_id": "F-01"}]}), encoding="utf-8")
        (work_dir / "integrity.json").write_text(
            json.dumps({"confidence": {"score": 0.5}, "grades": []}), encoding="utf-8"
        )
        (work_dir / "paper.md").write_text("正文", encoding="utf-8")

        context = ReviewContextLoader().load(str(work_dir))

        assert context.missing == []
        assert context.paper_text == "正文"
        assert context.findings == [{"finding_id": "F-01"}]
        assert context.confidence == 0.5

    def test_paper_file_path_resolves_its_directory(self, work_dir):
        (work_dir / "analysis.json").write_text(json.dumps({"findings": []}), encoding="utf-8")
        paper = work_dir / "paper.md"
        paper.write_text("正文", encoding="utf-8")

        context = ReviewContextLoader().load(str(paper))

        assert os.path.samefile(context.directory, str(work_dir))
        assert context.paper_text == "正文"
        assert context.available["analysis"] is True

    def test_unparsable_artifact_counts_as_missing(self, work_dir):
        (work_dir / "dataset.json").write_text("{not json", encoding="utf-8")

        context = ReviewContextLoader().load(str(work_dir))

        assert context.available["dataset"] is False
        assert "dataset.json" in context.missing


def test_severity_serialisation_keeps_labels():
    payload = _attack(SAMPLE_SIZE, MAJOR, finding_id="F-01").to_dict()

    assert payload["severity_label"] == "重大"
    assert payload["attack_label"] == "样本量攻击"


def _isolate(**overrides) -> ReviewContext:
    """只保留被测检查所需的产物，其余显式清空（避免其他攻击干扰断言）。"""

    values = {"findings": [], "analysis": {"analysis": {}}, "dataset": {}, "integrity": {}}
    values.update(overrides)
    return _rich_context(**values)


# ------------------------------------------------------------------ 样本量攻击


class TestSampleSizeAttacks:
    def test_small_sample_is_major(self):
        report = AdversarialReviewer().review(_rich_context())

        attack = next(item for item in report.attacks if item.finding_id == "F-05")
        assert attack.attack_type == SAMPLE_SIZE
        assert attack.severity == MAJOR

    def test_single_data_point_is_fatal_and_must_be_withdrawn(self):
        finding = _finding(
            "F-06", "相似度与成功率存在关系", metric="transfer_strength", category="transfer", sample_size=1
        )
        context = _isolate(findings=[finding], analysis={"analysis": {}, "findings": [finding]})

        report = AdversarialReviewer().review(context)
        attack = next(item for item in report.attacks if item.finding_id == "F-06")
        assert attack.severity == FATAL

        ranking = WeaknessRanker().rank(report)
        assert ranking.weaknesses[0].required_action == ACTION_WITHDRAW

    def test_large_sample_is_not_attacked(self):
        finding = _finding("F-02", "策略 x 平均收益 +10%", sample_size=40)
        context = _isolate(findings=[finding], analysis={"analysis": {}, "findings": [finding]})

        report = AdversarialReviewer().review(context)

        assert [item for item in report.attacks if item.attack_type == SAMPLE_SIZE] == []

    def test_conclusion_that_already_limits_itself_is_not_attacked_again(self):
        statement = "心智模型基于 12 个快照训练，但样本量不足以支撑准确率结论，本文不报告该指标"
        finding = _finding("F-08", statement, metric="prediction_accuracy", category="metacognition", sample_size=2)
        grade = _grade("F-08", statement, grade="C", category="metacognition")
        context = _isolate(
            findings=[finding],
            analysis={"analysis": {}, "findings": [finding]},
            integrity={"grades": [grade], "confidence": {"score": 0.6}, "provenance": {}},
        )

        report = AdversarialReviewer().review(context)

        assert report.attacks == []
        assert report.self_limited == ["F-08"]

    def test_missing_findings_records_a_skip(self):
        report = AdversarialReviewer().review(_isolate())

        assert any("样本量攻击" in note for note in report.skipped)
        assert SAMPLE_SIZE not in report.checks_run


# ------------------------------------------------------------------ 证据等级攻击


class TestWeakEvidenceAttacks:
    def test_grade_c_is_major_and_grade_b_is_minor(self):
        report = AdversarialReviewer().review(_rich_context())

        weak = {item.finding_id: item for item in report.attacks if item.attack_type == WEAK_EVIDENCE}
        assert weak["F-09"].severity == MAJOR
        assert weak["F-10"].severity == MINOR

    def test_finding_already_attacked_for_sample_size_is_not_attacked_twice(self):
        report = AdversarialReviewer().review(_rich_context())

        assert [
            item for item in report.attacks if item.finding_id == "F-05" and item.attack_type == WEAK_EVIDENCE
        ] == []

    def test_self_limited_statement_is_recorded_instead_of_accused(self):
        statement = "当前仅有 1 组配对观测，样本量不足以给出相关性结论"
        grade = _grade("F-06", statement, grade="C", category="transfer")
        context = _isolate(integrity={"grades": [grade], "confidence": {"score": 0.6}, "provenance": {}})

        report = AdversarialReviewer().review(context)

        assert report.attacks == []
        assert report.self_limited == ["F-06"]

    def test_missing_grades_records_a_skip(self):
        report = AdversarialReviewer().review(_isolate())

        assert any("证据等级攻击" in note for note in report.skipped)
        assert WEAK_EVIDENCE not in report.checks_run


# ------------------------------------------------------------------ 相关当因果


class TestCorrelationCausationAttacks:
    def test_correlational_finding_with_causal_wording_is_attacked(self):
        finding = _finding(
            "F-06", "相似度 0.8 带来更高的迁移成功率", metric="transfer_strength", category="transfer", sample_size=40
        )
        context = _isolate(findings=[finding], analysis={"analysis": {}, "findings": [finding]})

        report = AdversarialReviewer().review(context)

        assert [item.attack_type for item in report.attacks] == [CORRELATION_CAUSATION]
        assert report.attacks[0].severity == MAJOR

    def test_paper_sentence_mixing_correlation_and_causation_is_attacked(self):
        context = _isolate(paper_text="相似度与迁移成功率的相关性导致成功率提升。")

        report = AdversarialReviewer().review(context)

        assert [item.attack_type for item in report.attacks] == [CORRELATION_CAUSATION]
        assert report.attacks[0].target.startswith("正文")
        assert CORRELATION_CAUSATION in report.checks_run

    def test_correlation_wording_without_a_causal_verb_is_not_attacked(self):
        context = _isolate(paper_text="相似度与迁移成功率呈正相关，本文不做因果解释。")

        report = AdversarialReviewer().review(context)

        assert report.attacks == []
        assert CORRELATION_CAUSATION in report.checks_run

    def test_forward_reference_style_causal_claim_is_attacked(self):
        context = _isolate(paper_text="相似度提升使得迁移成功率上升；两者的相关性证明了策略有效。")

        report = AdversarialReviewer().review(context)

        assert len(report.attacks) == 2


# ------------------------------------------------------------------ 无对照组 / 幸存者偏差


class TestControlGroupAndSurvivorship:
    def test_strategy_claims_without_a_control_group_are_attacked(self):
        report = AdversarialReviewer().review(_rich_context())

        attack = next(item for item in report.attacks if item.attack_type == NO_CONTROL_GROUP)
        assert attack.severity == MAJOR
        assert "策略已被应用" in attack.evidence

    def test_control_group_check_is_skipped_when_no_strategy_claim_exists(self):
        report = AdversarialReviewer().review(_isolate())

        assert [item for item in report.attacks if item.attack_type == NO_CONTROL_GROUP] == []
        assert any("无对照组攻击" in note for note in report.skipped)

    def test_all_positive_strategy_benefits_trigger_survivorship(self):
        report = AdversarialReviewer().review(_rich_context())

        attack = next(item for item in report.attacks if item.attack_type == SURVIVORSHIP)
        assert attack.severity == MAJOR
        assert "failures=0" in attack.evidence

    def test_recorded_failure_silences_survivorship(self):
        dataset = {
            "strategies": [{"benefit": 0.1}, {"benefit": -0.3}],
            "similarity_pairs": [{"domain": "a"}],
            "phase_detection": {"commit_count": 53},
            "phases": [{}, {}],
        }

        report = AdversarialReviewer().review(_rich_context(dataset=dataset))

        assert [item for item in report.attacks if item.attack_type == SURVIVORSHIP] == []
        assert SURVIVORSHIP in report.checks_run

    def test_survivorship_is_skipped_without_strategy_observations(self):
        report = AdversarialReviewer().review(_isolate())

        assert any("幸存者偏差检查" in note for note in report.skipped)


# ------------------------------------------------------------------ 多重比较


class TestMultipleComparisons:
    def test_unadjusted_significance_is_flagged(self):
        analysis = {
            "analysis": {
                "success_rate_trend": {"p_value": 0.04},
                "similarity_correlation": {"p_value": 0.2},
                "comparisons": [{"label": "早半段 vs 晚半段", "p_value": 0.03}],
                "strategy_stats": [{"label": "a"}, {"label": "b"}],
            }
        }

        report = AdversarialReviewer().review(_isolate(analysis=analysis))

        attack = next(item for item in report.attacks if item.attack_type == MULTIPLE_COMPARISONS)
        assert attack.severity == MAJOR
        assert "Bonferroni" in attack.statement
        assert "tests=5" in attack.evidence

    def test_signal_surviving_the_correction_is_not_flagged(self):
        analysis = {
            "analysis": {
                "success_rate_trend": {"p_value": 0.0001},
                "similarity_correlation": {"p_value": 0.2},
                "comparisons": [{"label": "c", "p_value": 0.3}],
            }
        }

        report = AdversarialReviewer().review(_isolate(analysis=analysis))

        assert [item for item in report.attacks if item.attack_type == MULTIPLE_COMPARISONS] == []
        assert MULTIPLE_COMPARISONS in report.checks_run

    def test_no_p_values_records_a_skip(self):
        report = AdversarialReviewer().review(_isolate())

        assert any("多重比较" in note for note in report.skipped)
        assert MULTIPLE_COMPARISONS not in report.checks_run


# ------------------------------------------------------------------ 过度泛化


class TestOvergeneralization:
    def test_single_domain_transfer_evidence_is_attacked(self):
        report = AdversarialReviewer().review(_rich_context())

        transfer = [
            item for item in report.attacks if item.attack_type == OVERGENERALIZATION and item.category == "transfer"
        ]
        assert len(transfer) == 1
        assert transfer[0].severity == MAJOR

    def test_single_repository_evolution_claim_is_minor(self):
        report = AdversarialReviewer().review(_rich_context())

        evolution = [
            item for item in report.attacks if item.attack_type == OVERGENERALIZATION and item.category == "evolution"
        ]
        assert len(evolution) == 1
        assert evolution[0].severity == MINOR

    def test_three_domains_silence_the_transfer_attack(self):
        dataset = {
            "strategies": [{"benefit": 0.1}],
            "similarity_pairs": [{"domain": name} for name in ("a", "b", "c")],
            "phase_detection": {"commit_count": 0},
            "phases": [],
        }

        report = AdversarialReviewer().review(_rich_context(dataset=dataset))

        assert [
            item for item in report.attacks if item.attack_type == OVERGENERALIZATION and item.category == "transfer"
        ] == []

    def test_overgeneralization_is_skipped_without_a_dataset(self):
        report = AdversarialReviewer().review(_rich_context(dataset={}))

        assert any("过度泛化检查" in note for note in report.skipped)

    def test_audit_reports_every_check_that_actually_ran(self):
        dataset = {
            "strategies": [{"benefit": 0.1}, {"benefit": -0.1}],
            "similarity_pairs": [{"domain": name} for name in ("a", "b", "c")],
            "phase_detection": {"commit_count": 0},
            "phases": [],
        }
        findings = [_finding("F-02", "策略 x 平均收益 +10%", sample_size=40)]
        context = _rich_context(
            findings=findings,
            analysis={"analysis": {"success_rate_trend": {"p_value": 0.0001}}, "findings": findings},
            dataset=dataset,
            integrity={
                "grades": [_grade("F-02", findings[0]["statement"])],
                "confidence": {"score": 0.9},
                "provenance": {},
            },
            paper_text="本系统的自身演化过程。",
        )

        report = AdversarialReviewer().review(context)

        # 策略收益全为正已被上面的数据集排除；这里只应剩下“没有对照组”这一条
        assert [item.attack_type for item in report.attacks] == [NO_CONTROL_GROUP]
        assert report.is_clean is False
        assert CORRELATION_CAUSATION in report.checks_run
        assert report.skipped == []


# ------------------------------------------------------------------ 排序


class TestWeaknessRanker:
    def test_severity_order_and_critical_category_priority(self):
        report = AdversarialReviewer().review(_rich_context())

        ranking = WeaknessRanker().rank(report)

        severities = [item.severity for item in ranking.weaknesses]
        assert severities == sorted(severities, key=(FATAL, MAJOR, MINOR).index)
        assert ranking.weaknesses[0].category in ("transfer", "evolution")
        assert [item.rank for item in ranking.weaknesses] == list(range(1, len(ranking.weaknesses) + 1))
        assert ranking.severity_counts[MINOR] == 2
        assert len(ranking.affected_findings) == len(set(ranking.affected_findings))

    def test_actions_follow_the_public_table(self):
        report = AdversarialReviewer().review(_rich_context())

        ranking = WeaknessRanker().rank(report)
        actions = {item.attack_type: item.required_action for item in ranking.weaknesses}

        assert actions[SAMPLE_SIZE] == "downgrade"
        assert actions[NO_CONTROL_GROUP] == "reword"
        assert actions[OVERGENERALIZATION] == "scope"

    def test_fatal_always_means_withdraw(self):
        report = AdversarialReviewer().review(
            _isolate(
                findings=[_finding("F-01", "只有一个样本", sample_size=1)],
                analysis={"analysis": {}, "findings": [_finding("F-01", "只有一个样本", sample_size=1)]},
            )
        )

        ranking = WeaknessRanker().rank(report)

        assert ranking.weaknesses[0].severity == FATAL
        assert ranking.weaknesses[0].required_action == ACTION_WITHDRAW
        assert ranking.fatal_count == 1


# ------------------------------------------------------------------ 作者回应


class TestRebuttalGenerator:
    def test_overgeneralization_rule_rebuts_minor_only(self):
        generator = RebuttalGenerator()

        assert generator.generate(_attack(OVERGENERALIZATION, MINOR)).stance == REBUT
        assert generator.generate(_attack(OVERGENERALIZATION, MAJOR)).stance == PARTIAL

    def test_major_attack_returning_a_rebuttal_is_forced_to_accept(self, monkeypatch):
        generator = RebuttalGenerator()
        monkeypatch.setattr(generator, "_respond", lambda attack: (REBUT, "反驳", REVISION_REWORD))

        result = generator.generate(_attack(NO_CONTROL_GROUP, MAJOR))

        assert result.stance == ACCEPT
        assert result.revision == REVISION_DOWNGRADE

    def test_minor_attack_may_keep_its_rebuttal(self, monkeypatch):
        generator = RebuttalGenerator()
        monkeypatch.setattr(generator, "_respond", lambda attack: (REBUT, "反驳", REVISION_REWORD))

        result = generator.generate(_attack(OVERGENERALIZATION, MINOR))

        assert result.stance == REBUT
        assert result.revision == REVISION_REWORD

    def test_fatal_sample_size_attack_always_withdraws(self):
        result = RebuttalGenerator().generate(_attack(SAMPLE_SIZE, FATAL, finding_id="F-06", evidence="sample_size=1"))

        assert result.stance == ACCEPT
        assert result.revision == REVISION_WITHDRAW


# ------------------------------------------------------------------ 论文修订


class TestPaperRevisionEngine:
    @staticmethod
    def _pipeline(context):
        report = AdversarialReviewer().review(context)
        ranking = WeaknessRanker().rank(report)
        rebuttals = RebuttalGenerator().generate_all(ranking.weaknesses)
        engine = PaperRevisionEngine()
        return report, ranking, rebuttals, engine, engine.revise(context, ranking, rebuttals)

    def test_revision_never_raises_confidence(self):
        *_, result = self._pipeline(_rich_context())

        assert result.baseline_confidence == 0.9
        assert result.revised_confidence < result.baseline_confidence

    def test_fatal_attack_withdraws_the_conclusion_and_counts_as_critical(self):
        finding = _finding(
            "F-06", "相似度与成功率存在关系", metric="transfer_strength", category="transfer", sample_size=1
        )
        context = _isolate(
            findings=[finding],
            analysis={"analysis": {}, "findings": [finding]},
            integrity={
                "grades": [_grade("F-06", finding["statement"], grade="A", category="transfer")],
                "confidence": {"score": 0.9},
                "provenance": {},
            },
        )

        *_, result = self._pipeline(context)

        assert result.withdrawn_findings == ["F-06"]
        assert {item["finding_id"]: item["grade"] for item in result.revised_grades} == {"F-06": "C"}
        assert result.critical_count >= 1

    def test_global_attack_spreads_within_its_category_only(self):
        finding = _finding("F-02", "策略 x 平均收益 +46.9%", sample_size=20)
        context = _rich_context(
            findings=[finding],
            analysis={"analysis": {}, "findings": [finding]},
            dataset={"strategies": [{"benefit": 0.1}]},
            integrity={
                "grades": [
                    _grade("F-02", finding["statement"], grade="A", category="strategy"),
                    _grade("F-01", "总体成功率上升", grade="A", category="capability"),
                ],
                "confidence": {"score": 0.9},
                "provenance": {},
            },
        )

        *_, result = self._pipeline(context)

        grades = {item["finding_id"]: item["grade"] for item in result.revised_grades}
        assert grades["F-02"] == "B"
        assert grades["F-01"] == "A"

    def test_other_category_never_spreads(self):
        grades = {"F-01": {"grade": "A", "category": "other"}}
        weakness = RankedWeakness(
            rank=1,
            attack_id="A-01",
            attack_type=NO_CONTROL_GROUP,
            severity=MAJOR,
            required_action=ACTION_REWORD,
            finding_id="",
            category="other",
            target="全局",
            statement="意见",
            evidence="证据",
            suggested_fix="修改",
        )

        assert PaperRevisionEngine._targets(weakness, grades) == []

    def test_markdown_revision_annotates_findings_and_inserts_the_audit_section(self):
        text = (
            "# 论文\n\n## 5 结果与分析\n\n"
            "- F-02 策略 x 平均收益 +46.9%（证据等级 A）\n"
            "- F-01 总体成功率上升（证据等级 A）\n\n"
            "## 参考文献\n\n[1] 某文献\n"
        )
        context = _rich_context()

        report, ranking, rebuttals, engine, result = self._pipeline(context)
        section = engine.audit_section(context, report, ranking, rebuttals, result)
        revised = engine.revise_markdown(text, result, audit_section=section)

        # 行内等级必须同步，否则正文仍声称 A 级证据
        finding_lines = {line.split()[1]: line for line in revised.splitlines() if line.startswith("- F-")}
        assert "（证据等级 A → C）" in finding_lines["F-02"]
        assert "（证据等级 A）" in finding_lines["F-01"]
        assert "〔审稿后降级：" in revised
        assert revised.index("## 审稿意见与作者回应") < revised.index("## 参考文献")
        assert "| 编号 | 类型 | 严重度 | 对象 | 审稿意见 | 作者回应 | 论文修订 |" in revised
        assert "| A-01 | 样本量攻击 | 重大 | F-02 |" in revised

    def test_markdown_revision_is_idempotent(self):
        text = "# 论文\n\n## 5 结果与分析\n\n- F-02 策略 x 平均收益 +46.9%（证据等级 A）\n\n## 参考文献\n\n[1] 某文献\n"
        context = _rich_context()

        report, ranking, rebuttals, engine, result = self._pipeline(context)
        section = engine.audit_section(context, report, ranking, rebuttals, result)
        once = engine.revise_markdown(text, result, audit_section=section)
        twice = engine.revise_markdown(once, result, audit_section=section)

        # 重复审计不能把标注叠加成一串，也不能把审计章节插两次
        assert twice == once
        assert twice.count("## 审稿意见与作者回应（对抗性审计）") == 1

    def test_missing_baseline_confidence_is_reported_not_assumed(self):
        context = _isolate(
            findings=[_finding("F-02", "策略 x 平均收益 +46.9%", sample_size=2)],
            analysis={"analysis": {}, "findings": [_finding("F-02", "策略 x 平均收益 +46.9%", sample_size=2)]},
            integrity={
                "grades": [_grade("F-02", "策略 x 平均收益 +46.9%", grade="A")],
                "provenance": {},
            },
        )

        *_, result = self._pipeline(context)

        assert result.baseline_confidence is None
        assert result.confidence_delta == 0.0
        assert any("没有原置信度" in note for note in result.notes)


# ------------------------------------------------------------------ CLI 端到端


class TestAdversarialReviewCli:
    def test_end_to_end_keeps_a_backup_and_never_raises_confidence(self, work_dir, capsys):
        module = _import_autotestgen()
        paper = work_dir / "paper.md"
        module.run_self_research(str(paper), "markdown", str(work_dir), integrity_report=True)
        capsys.readouterr()
        before = paper.read_text(encoding="utf-8")

        module.run_adversarial_review(str(paper), None, str(work_dir))

        after = paper.read_text(encoding="utf-8")
        assert (work_dir / "paper.pre_review.md").read_text(encoding="utf-8") == before
        assert after != before
        assert "## 审稿意见与作者回应（对抗性审计）" in after
        assert "实际执行的检查" in after

        payload = json.loads((work_dir / "adversarial_review.json").read_text(encoding="utf-8"))
        revision = payload["revision"]
        assert revision["revised_confidence"] <= revision["baseline_confidence"]
        assert payload["audit"]["attack_count"] == len(payload["audit"]["attacks"])
        assert set(payload["ranking"]["severity_counts"]) == {FATAL, MAJOR, MINOR}
        assert len(payload["rebuttals"]) == len(payload["ranking"]["weaknesses"])
        assert "AdversarialReviewer" in capsys.readouterr().out

    def test_missing_artifacts_are_reported_instead_of_invented(self, work_dir, capsys):
        module = _import_autotestgen()
        (work_dir / "paper.md").write_text("# 只有正文\n", encoding="utf-8")

        module.run_adversarial_review(str(work_dir), None, str(work_dir))

        payload = json.loads((work_dir / "adversarial_review.json").read_text(encoding="utf-8"))
        assert payload["audit"]["attack_count"] == 0
        assert payload["audit"]["attacks"] == []

        skipped = " ".join(payload["audit"]["skipped"])
        for name in ("dataset.json", "analysis.json", "integrity.json"):
            assert name in skipped

        output = capsys.readouterr().out
        assert "[跳过]" in output
        assert "未发现可攻击的漏洞" in output
