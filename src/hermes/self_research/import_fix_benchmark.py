"""用真实历史里「人类补 import」的配对，测 oracle 的**来源一致率**与**放弃率**。

与语义缺陷基准不同，``undefined_name`` 这一类的判据可以做到完全干净：父状态真的缺一个
名字（由 pyflakes 判定，不是我们自己的感知器），人类当年真的补了 import（提交里的新增行）。
所以这里考的**不是**「能不能检出」—— AST 扫描谁都会，检出率天然接近 1，考的是**判断得对不对**：

============ ============================================================ ============
指标          含义                                                         期望
============ ============================================================ ============
检出率        系统的感知器看到了这条缺陷的比例                              ≈1.0
来源一致率    系统给的 import 来源 vs 人类当年给的（这是本基准的真正考点）      越高越好
放弃率        oracle 说「无法唯一确定」的比例                               越低越好，但不许猜
假阳性率      补丁落地后文件里新出现的未定义名（必须 0）                        0
============ ============================================================ ============

为什么「来源一致率」比「检出率」有信息量：检出率只说明系统看见了，一致率才说明它**判断对了**。
一个把 ``unicode`` 补成 ``from builtins import unicode`` 的系统，检出率满分，来源一致率零分。

两个已知的口径含糊处，都不藏起来，而是单列：

* 相对导入 vs 绝对导入——人类写 ``from ._compat import reraise``，系统给
  ``from flask._compat import reraise``，两者语义相同但模块名不同。严格一致的旁边另报
  「宽一致」（模块尾段相同），让人自己判断。
* 感知器没看到、但 oracle 仍给出答案的配对——仍参与来源一致率，但带 ``detected=False`` 标记。
"""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hermes.core.interfaces import DomainContext, PainPoint

from .import_fix_miner import ImportFixCase, import_bindings, mine_repository, undefined_names
from .real_defect_benchmark import GitRepo

DECISION_PATCH = "patch"
DECISION_ABSTAIN = "abstain"

AGREE_EXACT = "exact"
AGREE_LOOSE = "loose"
AGREE_SAME_PACKAGE = "same_package"
AGREE_DIFFERENT = "different"
AGREE_NONE = "none"


@dataclass
class PairOutcome:
    """一条配对上的完整判决。"""

    file: str
    name: str
    human_statement: str
    human_key: Tuple[str, str]
    detected: bool
    decision: str
    system_statement: Optional[str] = None
    system_key: Optional[Tuple[str, str]] = None
    confidence: Optional[str] = None
    agreement: str = AGREE_NONE
    false_positive: Optional[bool] = None
    reason: str = ""

    @property
    def agreed(self) -> bool:
        return self.agreement in (AGREE_EXACT, AGREE_LOOSE)

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "name": self.name,
            "human_statement": self.human_statement,
            "human_key": list(self.human_key),
            "detected": self.detected,
            "decision": self.decision,
            "system_statement": self.system_statement,
            "system_key": list(self.system_key) if self.system_key else None,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "false_positive": self.false_positive,
            "reason": self.reason,
        }


@dataclass
class ImportFixReport:
    repo: str = ""
    outcomes: List[PairOutcome] = field(default_factory=list)
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
    def patched(self) -> List[PairOutcome]:
        return [item for item in self.outcomes if item.decision == DECISION_PATCH]

    @property
    def abstain_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.outcomes if item.decision == DECISION_ABSTAIN), self.pairs)

    @property
    def agree_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.patched if item.agreed), len(self.patched))

    @property
    def strict_agree_rate(self) -> Optional[float]:
        return self._ratio(sum(1 for item in self.patched if item.agreement == AGREE_EXACT), len(self.patched))

    @property
    def same_package_rate(self) -> Optional[float]:
        """同名字、同包的不同层级（``flask`` vs ``flask.globals``）。**不算一致**，但和
        「指了个不相干的模块」不是一回事，所以单列。"""

        return self._ratio(sum(1 for item in self.patched if item.agreement == AGREE_SAME_PACKAGE), len(self.patched))

    @property
    def wrong_module_rate(self) -> Optional[float]:
        """指了个既非同一模块、也非同一包的名字 —— 这是真正该被追究的那一类。"""

        return self._ratio(sum(1 for item in self.patched if item.agreement == AGREE_DIFFERENT), len(self.patched))

    @property
    def decided_agree_rate(self) -> Optional[float]:
        """把自己放弃的那部分也算进分母：**放弃不算猜对**。"""

        return self._ratio(sum(1 for item in self.outcomes if item.agreed), self.pairs)

    @property
    def false_positive_rate(self) -> Optional[float]:
        graded = [item for item in self.patched if item.false_positive is not None]
        return self._ratio(sum(1 for item in graded if item.false_positive), len(graded))

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo": self.repo,
            "pairs": self.pairs,
            "detection_rate": self.detection_rate,
            "abstain_rate": self.abstain_rate,
            "agree_rate": self.agree_rate,
            "strict_agree_rate": self.strict_agree_rate,
            "same_package_rate": self.same_package_rate,
            "wrong_module_rate": self.wrong_module_rate,
            "decided_agree_rate": self.decided_agree_rate,
            "false_positive_rate": self.false_positive_rate,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": self.notes,
        }

    def report_line(self) -> str:
        return (
            f"{self.repo}：{self.pairs} 条配对  "
            f"检出率={_fmt(self.detection_rate)}  来源一致率={_fmt(self.agree_rate)}"
            f"（严格 {_fmt(self.strict_agree_rate)}）  同包不同层={_fmt(self.same_package_rate)}  "
            f"指错模块={_fmt(self.wrong_module_rate)}  放弃率={_fmt(self.abstain_rate)}  "
            f"把放弃算进分母的一致率={_fmt(self.decided_agree_rate)}  "
            f"假阳性率={_fmt(self.false_positive_rate)}"
        )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _splice(source: str, after_line: int, statement: str) -> str:
    """按 oracle 说的插入点把 import 行拼进去（只在内存里，不动磁盘）。"""

    lines = source.splitlines(keepends=True)
    index = max(0, min(after_line, len(lines)))
    return "".join(lines[:index]) + statement.strip() + "\n" + "".join(lines[index:])


def _synthesize(case: ImportFixCase) -> PainPoint:
    """感知器没看到这条缺陷时，仍然造一个等价 pain point —— 好让 oracle 被考到，
    同时把 ``detected=False`` 如实记下来（否则「没检出」会伪装成「放弃」）。"""

    return PainPoint(
        id=f"{case.file}:{case.human_line}:undefined_name",
        type="undefined_name",
        severity="major",
        message=f"使用了从未定义的名字 {case.name!r}（{case.file}:{case.human_line}）",
        location=case.file,
        context={
            "line": case.human_line,
            "column": 0,
            "file": case.file,
            "check": "undefined_name",
            "name": case.name,
        },
    )


def _judge(case: ImportFixCase, plan, parent_source: str) -> PairOutcome:
    operations = getattr(plan, "operations", None) or []
    if not operations:
        return PairOutcome(
            file=case.file,
            name=case.name,
            human_statement=case.human_statement,
            human_key=case.human_key,
            detected=False,
            decision=DECISION_ABSTAIN,
            reason=(getattr(plan, "description", "") or "").strip(),
        )

    operation = operations[0]
    statement = operation.get("statement") or operation.get("lines", [""])[0]
    key = import_bindings(statement).get(case.name)
    after_line = int(operation.get("after_line") or 0)

    before = {name for _, name in undefined_names(parent_source, case.file)}
    after = {name for _, name in undefined_names(_splice(parent_source, after_line, statement), case.file)}

    return PairOutcome(
        file=case.file,
        name=case.name,
        human_statement=case.human_statement,
        human_key=case.human_key,
        detected=False,
        decision=DECISION_PATCH,
        system_statement=statement,
        system_key=key,
        confidence=operation.get("confidence"),
        agreement=_agreement(key, case.human_key),
        false_positive=bool(after - before),
        reason=operation.get("basis", ""),
    )


def _agreement(system: Optional[Tuple[str, str]], human: Tuple[str, str]) -> str:
    if system is None:
        # 系统插入的 import 绑定的不是那个名字 —— 那它根本没修这条缺陷
        return AGREE_DIFFERENT
    if system == human:
        return AGREE_EXACT
    system_module, system_name = system
    human_module, human_name = human
    if not system_name or system_name != human_name:
        return AGREE_DIFFERENT
    if not system_module or not human_module:
        # 一边是整个模块（`import json`）、一边是带去处的符号（`from helpers import json`）：
        # 没有可比的两段模块名。空串是 `endswith` 的万能匹配，必须挡住，否则一致率会被虚高。
        return AGREE_DIFFERENT
    if system_module.endswith(human_module) or human_module.endswith(system_module):
        # `from ._compat import x` 与 `from flask._compat import x`：语义相同，模块名不同
        return AGREE_LOOSE
    if system_module.startswith(human_module + ".") or human_module.startswith(system_module + "."):
        # `import importlib` vs `import importlib.util`；`from flask import x` vs
        # `from flask.globals import x`：同包的不同层级。可能是重导出，也可能不够用。
        return AGREE_SAME_PACKAGE
    return AGREE_DIFFERENT


class ParentTree:
    """把某个提交的父状态 checkout 到临时 worktree。

    oracle 的第 3 层（项目内唯一导出符号）要扫整个项目，所以不能只喂一个文件 ——
    只喂文件会让所有项目内符号都「找不到」，把放弃率人为抬高。

    路径必须**绝对**：``git -C <repo> worktree add <相对路径>`` 是相对那个仓库解析的，
    传相对路径会让 checkout 跑到别处、而这里拿到的目录空空如也 —— 那样整套指标会静默
    退化成「全是放弃」，正是本项目最该避免的那种假报告。所以 ``open`` 会当场验证目录里
    真的有 ``.py`` 文件，否则返回 None。
    """

    def __init__(self, repo: GitRepo, root: str):
        self.repo = repo
        self.root = os.path.abspath(root)

    def open(self, revision: str) -> Optional[str]:
        destination = os.path.join(self.root, revision[:12])
        shutil.rmtree(destination, ignore_errors=True)
        try:
            self.repo.run("worktree", "add", "--detach", "--force", destination, revision)
        except Exception:
            return None
        if not _has_python_files(destination):
            self.close(destination)
            return None
        return destination

    def close(self, destination: Optional[str]) -> None:
        if not destination:
            return
        try:
            self.repo.run("worktree", "remove", "--force", destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
        try:
            self.repo.run("worktree", "prune")
        except Exception:
            pass


def evaluate_repository(
    repo_path: str,
    *,
    limit: Optional[int] = None,
    max_pairs: int = 150,
    work_root: Optional[str] = None,
    git=None,
) -> ImportFixReport:
    """挖配对，然后拿 oracle 逐条对答案。"""

    from hermes.plugins.default_autotestgen.real_perceiver import RealPerceiver
    from hermes.plugins.default_autotestgen.real_sage import RealSage

    repo = git or GitRepo(repo_path)
    mine = mine_repository(repo_path, limit=limit, max_cases=max_pairs, git=repo)
    report = ImportFixReport(repo=repo_path)
    report.notes.extend(mine.notes)

    cases = mine.clean[:max_pairs]
    if not cases:
        report.notes.append("没有干净的配对，无法测来源一致率")
        return report

    root = work_root or os.path.join(os.path.dirname(os.path.abspath(repo_path)), "_worktrees")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    trees = ParentTree(repo, root)
    perceiver = RealPerceiver()
    sage = RealSage()

    grouped: Dict[str, List[ImportFixCase]] = {}
    for case in cases:
        grouped.setdefault(case.parent, []).append(case)

    for parent, group in grouped.items():
        tree = trees.open(parent)
        if tree is None:
            report.notes.append(
                f"无法 checkout 父状态 {parent[:8]}（或 checkout 出来的目录里没有 .py 文件），"
                f"跳过它下面的 {len(group)} 条配对 —— 这不是「放弃」，是不该计入"
            )
            continue
        try:
            context = DomainContext(domain="python", path=tree)
            detected = _detect(perceiver, context)
            for case in group:
                point = detected.get((case.file, case.name))
                outcome = _judge(case, sage.generate_patch(context, point or _synthesize(case)), _read(tree, case.file))
                outcome.detected = point is not None
                report.outcomes.append(outcome)
        finally:
            trees.close(tree)
    return report


def _detect(perceiver, context: DomainContext) -> Dict[Tuple[str, Optional[str]], PainPoint]:
    """跑感知器。老仓库在新解释器下会刷一屏 ``SyntaxWarning``，那是环境差异不是结论，
    在这里挡掉，免得把报告淹了。"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = perceiver.detect(context)
    return {
        (point.location, (point.context or {}).get("name")): point for point in points if point.type == "undefined_name"
    }


def _has_python_files(path: str) -> bool:
    for _, _, files in os.walk(path):
        if any(name.endswith(".py") for name in files):
            return True
    return False


def _read(tree: str, relative: str) -> str:
    path = os.path.join(tree, relative.replace("/", os.sep))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return ""
