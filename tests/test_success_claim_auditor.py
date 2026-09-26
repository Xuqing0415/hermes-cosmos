"""成功声明审计：判据不足的「成功」不许当结论。

验收口径（来自修复方案）：

  * 批次 1 修复前的 run_adversarial_fl（``if acc_drop < 10: print(" SUCCESS: ...")``）
    必须被报成 ``UNVERIFIED_CLAIM``；
  * 修复后的 run_adversarial_fl 必须报 OK——这里直接审计仓库里那个真实文件，
    所以这条断言是回归钉子：谁把判据删回一条，它就红。
"""

from pathlib import Path

from hermes.self_research.data_provenance import DataProvenanceTagger
from hermes.self_research.integrity_checker import CLAIM_SECTION, IntegrityChecker
from hermes.self_research.paper_writer import PaperWriter
from hermes.self_research.research_data_collector import ResearchDataset
from hermes.self_research.success_claim_auditor import (
    MIN_CRITERIA,
    VERDICT_UNVERIFIED,
    VERDICT_VERIFIED,
    ClaimAuditReport,
    SuccessClaimAuditor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_TARGET = REPO_ROOT / "hermes_unified" / "federated" / "run_adversarial_fl.py"

# 五个数据源都在，采集层才不产 warning——这样 is_clean 只由「成功声明」决定。
ALL_SOURCES = {
    "snapshot_db": True,
    "notification_log": True,
    "policy": True,
    "knowledge_graph": True,
    "git_history": True,
}

# 逐字摘自修复前（commit 44f57a2）的 report 尾巴：成功判据只有「掉点少于 10」这一条。
PRE_FIX_SOURCE = """
def run_adversarial_experiment(baseline_acc, final_acc, attack_events):
    acc_drop = (baseline_acc - final_acc) * 100
    print(f"Number of Attack Events: {len(attack_events)}")

    print("\\n OUTCOME")
    if acc_drop < 10:
        print(" SUCCESS: Defense effectively mitigated attacks!")
    elif acc_drop < 30:
        print("  PARTIAL SUCCESS: Some accuracy loss but model usable")
    else:
        print(" FAILURE: Attacks overwhelmed the defense")

    print("=" * 60)

    return {"acc_drop": acc_drop}
"""


def audit(source: str, filename: str = "sample.py"):
    return SuccessClaimAuditor().audit_source(source, filename)


def clean_provenance():
    return DataProvenanceTagger().tag(ResearchDataset(sources=dict(ALL_SOURCES)))


class TestPreFixCounterexample:
    """验收的反面：修复前的写法必须被抓住。"""

    def test_pre_fix_target_is_reported_as_unverified_claim(self):
        claims = audit(PRE_FIX_SOURCE, "run_adversarial_fl.py")

        assert claims, '修复前的 print(" SUCCESS: ...") 没有被认出来'
        assert all(claim.verdict == VERDICT_UNVERIFIED for claim in claims)
        assert all(len(claim.criteria) < MIN_CRITERIA for claim in claims)

        statements = " ".join(claim.statement for claim in claims)
        assert "SUCCESS: Defense effectively mitigated attacks!" in statements
        assert all(claim.function == "run_adversarial_experiment" for claim in claims)

    def test_pre_fix_reason_names_the_criteria_that_were_counted(self):
        claim = audit(PRE_FIX_SOURCE)[0]

        assert "acc_drop < 10" in " ".join(item.test for item in claim.criteria)
        assert "只有 1 条判据" in claim.reasons[0]

    def test_partial_success_branch_is_also_a_claim(self):
        claims = audit(PRE_FIX_SOURCE)

        assert any("PARTIAL SUCCESS" in claim.statement for claim in claims)


class TestFixedTarget:
    """验收的正面：修复后的真实文件必须通过。"""

    def test_fixed_target_has_no_unverified_claim(self):
        report = SuccessClaimAuditor(roots=[str(FIXED_TARGET)]).audit_paths()

        assert report.claims, "审计器没认出修复后的成功声明，这条验收就没有意义"
        assert report.is_clean, [claim.to_dict() for claim in report.unverified]

    def test_fixed_claim_is_backed_by_the_three_conditions(self):
        claims = SuccessClaimAuditor().audit_source(FIXED_TARGET.read_text(encoding="utf-8"))
        claim = next(item for item in claims if item.function == "classify_defense_result")

        assert claim.verdict == VERDICT_VERIFIED
        assert len(claim.criteria) == 5
        tests = " ".join(item.test for item in claim.criteria)
        assert "baseline_acc < chance" in tests
        assert "attack_events == 0" in tests
        assert "defended_acc < baseline_acc" in tests


class TestFalsePositives:
    """宁可不报，不可误报：这些都不算成功声明。"""

    def test_comparison_operand_is_not_a_claim(self):
        source = 'def show(verdict):\n    if verdict == "SUCCESS":\n        print("ok")\n'

        assert audit(source) == []

    def test_ternary_rendering_is_not_a_claim(self):
        source = 'def show(ok):\n    print("SUCCESS" if ok else "FAILED")\n'

        assert audit(source) == []

    def test_attribute_ternary_rendering_is_not_a_claim(self):
        source = "def show(result):\n    print(f\"status: {'SUCCESS' if result.success else 'FAILED'}\")\n"

        assert audit(source) == []

    def test_dict_payload_is_not_a_claim(self):
        source = "def audit_log():\n    return {'status': 'SUCCESS', 'detail': 'sample'}\n"

        assert audit(source) == []

    def test_docstring_mentioning_success_is_not_a_claim(self):
        source = 'def documented():\n    """Returns SUCCESS when everything is fine."""\n    return True\n'

        assert audit(source) == []

    def test_failure_return_is_not_a_claim(self):
        source = 'def classify(a):\n    if a < 1:\n        return "INCONCLUSIVE", ["a"]\n    return "SUCCESS", []\n'
        claims = audit(source)

        assert len(claims) == 1
        assert "SUCCESS" in claims[0].statement


class TestCriteriaCounting:
    def test_three_rejection_guards_verify_a_tail_return(self):
        source = (
            "def classify(baseline, attacks):\n"
            "    reasons = []\n"
            "    if baseline < 0.1:\n"
            "        reasons.append('no baseline')\n"
            "    if attacks == 0:\n"
            "        reasons.append('no attacks')\n"
            "    if reasons:\n"
            "        return 'INCONCLUSIVE', reasons\n"
            "    return 'SUCCESS', []\n"
        )
        claim = audit(source)[0]

        assert len(claim.criteria) == 3
        assert claim.verdict == VERDICT_VERIFIED

    def test_two_rejection_guards_are_not_enough(self):
        source = (
            "def classify(baseline, attacks):\n"
            "    if baseline < 0.1:\n"
            "        return 'INCONCLUSIVE', ['no baseline']\n"
            "    if attacks == 0:\n"
            "        return 'INCONCLUSIVE', ['no attacks']\n"
            "    return 'SUCCESS', []\n"
        )
        claim = audit(source)[0]

        assert len(claim.criteria) == 2
        assert claim.verdict == VERDICT_UNVERIFIED

    def test_and_conjuncts_count_separately(self):
        source = (
            "def classify(a, b):\n"
            "    if a < 0.1 and b == 0 and a + b < 1:\n"
            "        return 'INCONCLUSIVE', []\n"
            "    return 'SUCCESS', []\n"
        )
        claim = audit(source)[0]

        assert len(claim.criteria) == 3
        assert claim.verdict == VERDICT_VERIFIED

    def test_nested_enclosing_guards_count_separately(self):
        source = (
            "def emit(a, b):\n"
            "    if a < 0.1:\n"
            "        if b == 0:\n"
            "            if a + b < 1:\n"
            "                print('SUCCESS')\n"
        )
        claim = audit(source)[0]

        assert len(claim.criteria) == 3
        assert all(item.kind == "enclosing" for item in claim.criteria)

    def test_unconsulted_accumulator_does_not_count(self):
        source = (
            "def classify(x):\n"
            "    notes = []\n"
            "    if x < 0:\n"
            "        notes.append('negative')\n"
            "    return 'SUCCESS', []\n"
        )
        claim = audit(source)[0]

        assert claim.criteria == ()
        assert claim.verdict == VERDICT_UNVERIFIED

    def test_unrelated_ifs_far_above_do_not_count(self):
        source = (
            "def classify(x):\n"
            "    if x > 1:\n"
            "        print('warmup')\n"
            "    if x > 2:\n"
            "        print('warmup')\n"
            "    if x > 3:\n"
            "        print('warmup')\n"
            "    work = x\n"
            "    return 'SUCCESS', []\n"
        )
        claim = audit(source)[0]

        assert claim.criteria == ()
        assert claim.verdict == VERDICT_UNVERIFIED

    def test_module_level_claim_has_no_criteria(self):
        claim = audit('print("SUCCESS")\n')[0]

        assert claim.function == "<module>"
        assert claim.criteria == ()
        assert claim.verdict == VERDICT_UNVERIFIED

    def test_report_success_call_is_a_claim(self):
        claim = audit("def do_work(x):\n    report_success()\n")[0]

        assert claim.kind == "report_call"
        assert claim.verdict == VERDICT_UNVERIFIED


class TestReport:
    def test_summary_line_distinguishes_clean_from_unverified(self):
        clean = SuccessClaimAuditor().audit_paths([str(FIXED_TARGET)])
        assert "审计通过" in clean.summary_line()

        dirty = ClaimAuditReport(claims=audit(PRE_FIX_SOURCE, "run_adversarial_fl.py"))
        assert "未经证成" in dirty.summary_line()

        empty = ClaimAuditReport()
        assert "未发现" in empty.summary_line()

    def test_missing_root_is_reported_not_faked(self):
        report = SuccessClaimAuditor().audit_paths([str(REPO_ROOT / "does_not_exist")])

        assert report.claims == []
        assert report.scanned_files == 0
        assert report.skipped_roots == [str(REPO_ROOT / "does_not_exist")]
        assert "不存在" in report.note

    def test_dict_form_carries_everything_the_paper_needs(self):
        report = SuccessClaimAuditor().audit_paths([str(FIXED_TARGET)])
        payload = report.to_dict()

        assert payload["is_clean"] is True
        assert payload["unverified_claims"] == []
        assert payload["claims"][0]["criteria_count"] >= MIN_CRITERIA
        assert payload["claims"][0]["claim_id"].startswith("claim:")


class TestIntegrityIntegration:
    _checker = IntegrityChecker()

    def test_unverified_claim_becomes_a_disclaimer(self):
        claims = SuccessClaimAuditor().audit_source(PRE_FIX_SOURCE, "run_adversarial_fl.py")
        audit_report = ClaimAuditReport(claims=claims)
        report = self._checker.check([], clean_provenance(), claim_audit=audit_report)

        assert report.unverified_claims
        assert report.is_clean is False
        disclaimers = [item for item in report.disclaimers if item.target == CLAIM_SECTION]
        assert disclaimers
        assert all(item.severity == "caution" for item in disclaimers)
        assert "未经证成" in disclaimers[0].text
        assert "run_adversarial_fl.py" in disclaimers[0].text
        assert report.claim_audit["claims"]

    def test_verified_claims_leave_the_report_clean(self):
        audit_report = SuccessClaimAuditor(roots=[str(FIXED_TARGET)]).audit_paths()
        report = self._checker.check([], clean_provenance(), claim_audit=audit_report)

        assert report.unverified_claims == []
        assert report.disclaimers == []
        assert report.is_clean is True
        assert "成功声明审计通过" in IntegrityChecker.summary_line(report)

    def test_omitting_the_audit_keeps_the_old_behaviour(self):
        report = self._checker.check([], clean_provenance())

        assert report.claim_audit == {}
        assert report.unverified_claims == []
        assert report.is_clean is True
        assert "成功声明审计" not in IntegrityChecker.summary_line(report)

    def test_paper_section_renders_the_unverified_claim(self):
        claims = SuccessClaimAuditor().audit_source(PRE_FIX_SOURCE, "run_adversarial_fl.py")
        report = self._checker.check([], clean_provenance(), claim_audit=ClaimAuditReport(claims=claims))

        section = PaperWriter()._integrity_section({"integrity": report.to_dict()})

        assert "6.6 成功声明审计" in section
        assert "[未经证成]" in section
        assert "run_adversarial_fl.py" in section

    def test_paper_section_marks_verified_claims(self):
        report = self._checker.check(
            [],
            clean_provenance(),
            claim_audit=SuccessClaimAuditor(roots=[str(FIXED_TARGET)]).audit_paths(),
        )

        section = PaperWriter()._integrity_section({"integrity": report.to_dict()})

        assert "6.6 成功声明审计" in section
        assert "[已证成]" in section
        assert "[未经证成]" not in section
