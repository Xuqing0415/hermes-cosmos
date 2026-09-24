"""拿「人类删掉未使用 import」的真实配对，测系统在**减法**这一侧的现状。

与 ``undefined_name`` 那边一样，判据是第三方事实（pyflakes 说没用过 + 提交真的删了它）。
不一样的是这里要问的三个问题：

1. **我们的感知器看得见吗** —— 检出率。把 ``RealPerceiver`` 落在同一个 (文件, 行) 上
   才算看得见。
2. **oracle 能动手吗** —— 目前 ``RealSage`` 只有「补 import」的模板，没有「删 import」的，
   所以预期是全部拒绝。这条要如实测出来，而不是靠读代码断言。
3. **名字交出来了吗** —— Sage 靠 ``PainPoint.context["name"]`` 决定动哪个名字（上一轮定下的
   规矩：不去解析 message）。如果感知器报 unused 时不带这个名字，那即便加了删除模板，
   oracle 也无从下手。这一条单列成 ``name_exposed_rate``。

运行期的那一关（删完之后测试是否还是绿的）在这里**测不了**：requests / flask 这些仓库的
测试在 Python 3.14 下连收集都跑不起来。所以本模块不产出假阳性率，只在报告里写明它缺什么 ——
「没测」和「测得 0」必须分开。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hermes.core.interfaces import DomainContext, PainPoint

from .real_defect_benchmark import GitRepo
from .unused_import_miner import RemovedImportCase, mine_repository

DECISION_PATCH = "patch"
DECISION_REFUSE = "refuse"


@dataclass
class RemovalOutcome:
    file: str
    line: int
    names: List[str]
    statement: str
    detected: bool
    name_exposed: bool
    decision: str
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "names": list(self.names),
            "statement": self.statement,
            "detected": self.detected,
            "name_exposed": self.name_exposed,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass
class UnusedBenchmarkReport:
    repo: str = ""
    outcomes: List[RemovalOutcome] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def pairs(self) -> int:
        return len(self.outcomes)

    def _ratio(self, numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 4) if denominator else None

    @property
    def detection_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.outcomes if item.detected), self.pairs)

    @property
    def name_exposed_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.outcomes if item.name_exposed), self.pairs)

    @property
    def refusal_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.outcomes if item.decision == DECISION_REFUSE), self.pairs)

    @property
    def action_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.outcomes if item.decision == DECISION_PATCH), self.pairs)

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo": self.repo,
            "pairs": self.pairs,
            "detection_rate": self.detection_rate,
            "name_exposed_rate": self.name_exposed_rate,
            "refusal_rate": self.refusal_rate,
            "action_rate": self.action_rate,
            "runtime_verdict": None,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": self.notes,
        }

    def report_line(self) -> str:
        return (
            f"{self.repo}：{self.pairs} 条配对  检出率={_fmt(self.detection_rate)}  "
            f"名字结构化交出={_fmt(self.name_exposed_rate)}  拒绝率={_fmt(self.refusal_rate)}  "
            f"能动手={_fmt(self.action_rate)}  运行期假阳性率=未测"
        )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _detect(perceiver, context: DomainContext) -> Dict[Tuple[str, int], PainPoint]:
    """(文件, 行) → pain point。老代码在新解释器下会刷 SyntaxWarning，这里挡掉。"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = perceiver.detect(context)
    found: Dict[Tuple[str, int], PainPoint] = {}
    for point in points:
        if point.type != "unused_import":
            continue
        line = int((point.context or {}).get("line") or 0)
        found[(point.location or "", line)] = point
    return found


def _judge(case: RemovedImportCase, plan) -> RemovalOutcome:
    operations = getattr(plan, "operations", None) or []
    decision = DECISION_PATCH if operations else DECISION_REFUSE
    return RemovalOutcome(
        file=case.file,
        line=case.line,
        names=list(case.names),
        statement=case.statement,
        detected=False,
        name_exposed=False,
        decision=decision,
        reason=(getattr(plan, "description", "") or "").strip(),
    )


def evaluate_repository(
    repo_path: str,
    *,
    limit: Optional[int] = None,
    max_pairs: int = 150,
    work_root: Optional[str] = None,
    git=None,
) -> UnusedBenchmarkReport:
    from hermes.plugins.default_autotestgen.real_perceiver import RealPerceiver
    from hermes.plugins.default_autotestgen.real_sage import RealSage

    from .import_fix_benchmark import ParentTree

    repo = git or GitRepo(repo_path)
    mined = mine_repository(repo_path, limit=limit, max_cases=max_pairs * 3, git=repo)
    report = UnusedBenchmarkReport(repo=repo_path)
    report.notes.extend(mined.notes)
    report.notes.append("运行期假阳性率未测：这些仓库的测试在 Python 3.14 下跑不起来，删得对不对没有第三方裁决")

    cases = mined.clean[:max_pairs]
    if not cases:
        report.notes.append("没有干净的配对，测不了")
        return report

    root = os.path.abspath(work_root or os.path.join(os.path.dirname(os.path.abspath(repo_path)), "_worktrees"))
    os.makedirs(root, exist_ok=True)
    trees = ParentTree(repo, root)
    perceiver = RealPerceiver()
    sage = RealSage()

    grouped: Dict[str, List[RemovedImportCase]] = {}
    for case in cases:
        grouped.setdefault(case.parent, []).append(case)

    for parent, group in grouped.items():
        tree = trees.open(parent)
        if tree is None:
            report.notes.append(f"无法 checkout 父状态 {parent[:8]}，跳过它下面的 {len(group)} 条配对")
            continue
        try:
            context = DomainContext(domain="python", path=tree)
            detected = _detect(perceiver, context)
            for case in group:
                point = detected.get((case.file, case.line))
                outcome = _judge(case, sage.generate_patch(context, point or _synthesize(case)))
                outcome.detected = point is not None
                outcome.name_exposed = bool((point.context or {}).get("name")) if point else False
                report.outcomes.append(outcome)
        finally:
            trees.close(tree)
    return report


def _synthesize(case: RemovedImportCase) -> PainPoint:
    """感知器没看见时也造一个等价 pain point，好把 oracle 的决策层单独考出来。"""

    return PainPoint(
        id=f"{case.file}:{case.line}:unused_import",
        type="unused_import",
        severity="minor",
        message=f"导入了但没用到的名字 {case.names}（{case.file}:{case.line}）",
        location=case.file,
        context={"line": case.line, "column": 0, "file": case.file, "check": "unused_import"},
    )
