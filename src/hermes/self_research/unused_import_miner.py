"""从一个真实仓库的历史里挖出「父状态某条 import 没被用到 → 人类把它删了」的配对。

这是 ``undefined_name`` 的**减法镜像**：那边是「缺了个名字，补上去」，这边是「多个名字，
删掉它」。判据同样是两个互相独立的第三方事实：

* 父状态那条 import 真的没被用到 —— 由 pyflakes 判定（不是我们自己的感知器）；
* 人类真的删了它 —— 由提交里**删除**的 import 行判定（只认删除行，否刚会把「新增 import」
  误当成「人类删了 import」）。

为什么值得单独做一类：加法和减法的能力边界不一样。补一个 import 要回答「这个名字从哪来」，
删一个 import 要回答「它是不是被别处用着」—— 而后者的三个大坑纯静态分析都看不穿：

1. **副作用**：``import my_pkg.plugins`` 可能只是为了让插件注册跑起来，删掉测试会静默变红；
2. **导出**：名字自己没被用到，但它是包的公开 API（``__init__.py`` 的再导出、动态 ``__all__``）；
3. **动态使用**：``eval`` / ``getattr`` / ``globals()`` 里用字符串拼出来的名字。

这三类只有跑起来才知道 —— 正好接上已经做好的 Knight。所以这一类的假阳性率是**运行期**
指标，不是语法层面的推断。

已知的污染陷阱，全部显式标记成 ``weak`` 而不是默默收下：

============ ================================================================
weak 原因      含义
============ ================================================================
package_init  被删的文件是 ``__init__.py``：删掉等于改包的公开 API
in_dunder_all 名字在该文件的 ``__all__`` 里（``__all__`` 是拼出来的才会被 pyflakes 漏掉）
dynamic_usage 文件里有 ``eval``/``getattr``/``globals``/``setattr``：名字可能被字符串拼出来
star_import   文件里有 ``from x import *``：unused 判断不可靠
dotted_side_effect ``import a.b``（没有 ``as``）：可能靠导入触发副作用
redefinition  人类删的是重复导入里的第一条（pyflakes 报的是 "redefinition of unused"）
============ ================================================================
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .import_fix_miner import IMPORT_LINE, _load_grader, _parse_quiet, _quiet, import_bindings

WEAK_PACKAGE_INIT = "package_init"
WEAK_DUNDER_ALL = "in_dunder_all"
WEAK_DYNAMIC_USAGE = "dynamic_usage"
WEAK_STAR_IMPORT = "star_import"
WEAK_DOTTED_SIDE_EFFECT = "dotted_side_effect"
WEAK_REDEFINITION = "redefinition"

UNUSED = "unused"
REDEFINITION = "redefinition"

STAR_IMPORT = re.compile(r"^from\s+[\w.]+\s+import\s+\*$")
# 注意：pyflakes 的 str(message) 带 ``文件:行:列:`` 前缀，所以不能加 ^...$ 锚点
UNUSED_NAME = re.compile(r"'([^']+)' imported but unused")
REDEFINED_NAME = re.compile(r"redefinition of unused '([^']+)' from line \d+")
DYNAMIC_CALLS = frozenset({"eval", "exec", "getattr", "setattr", "globals", "locals", "vars", "__import__"})
PY_SUFFIX = ".py"


@dataclass
class RemovedImportCase:
    """父状态在 ``file`` 第 ``line`` 行的那条 import 里，``names`` 都没人用，人类在 ``sha`` 里删了它。

    单位是**那一条语句**，不是每个绑定名：oracle 的动作是「删掉这一行」，一条语句绑十个
    没人用的名字仍然只该算一条样本，否则配对数会被一条删行放大十倍。
    """

    sha: str
    parent: str
    subject: str
    date: str
    file: str
    line: int
    names: List[str]
    statement: str
    weak: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.weak

    def to_dict(self) -> Dict[str, object]:
        return {
            "sha": self.sha,
            "parent": self.parent,
            "subject": self.subject,
            "date": self.date,
            "file": self.file,
            "line": self.line,
            "names": list(self.names),
            "statement": self.statement,
            "weak": self.weak,
            "clean": self.is_clean,
        }


@dataclass
class UnusedImportReport:
    repo: str = ""
    scanned_commits: int = 0
    candidate_commits: int = 0
    cases: List[RemovedImportCase] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def clean(self) -> List[RemovedImportCase]:
        return [case for case in self.cases if case.is_clean]

    @property
    def weak_reasons(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for case in self.cases:
            for reason in case.weak:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    @property
    def files(self) -> int:
        return len({case.file for case in self.clean})

    def summary_line(self) -> str:
        return (
            f"{self.repo}：扫描 {self.scanned_commits} 个提交，其中 {self.candidate_commits} 个删掉了 import；"
            f"得到 {len(self.cases)} 条配对（干净 {len(self.clean)} 条，分布在 {self.files} 个文件）"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo": self.repo,
            "scanned_commits": self.scanned_commits,
            "candidate_commits": self.candidate_commits,
            "pairs": len(self.cases),
            "clean_pairs": len(self.clean),
            "clean_files": self.files,
            "weak_reasons": self.weak_reasons,
            "cases": [case.to_dict() for case in self.cases],
            "notes": self.notes,
        }


def unused_import_names(source: str, filename: str) -> List[Tuple[int, str, str]]:
    """pyflakes 认为「导入了但没用到」的名字，返回 ``[(行号, 名字, 种类)]``。

    pyflakes 的消息里写的不是绑定名而是 ``module.name`` / 点号路径（``from typing import
    Dict`` 报的是 ``'typing.Dict'``，``import os.path`` 报的是 ``'os.path'``），所以这里只
    保留原文，由调用方按**行号**去和删掉的那条 import 对齐。
    """

    check, Reporter = _load_grader()
    found: List[Tuple[int, str, str]] = []

    class _Collector(Reporter):
        def unexpectedError(self, path, message):  # noqa: N802 - pyflakes 接口
            pass

        def syntaxError(self, path, message, lineno, offset, text):  # noqa: N802
            pass

        def flake(self, message):
            text = str(message)
            lineno = int(getattr(message, "lineno", 0) or 0)
            unused = UNUSED_NAME.search(text)
            if unused:
                found.append((lineno, unused.group(1), UNUSED))
                return
            redefined = REDEFINED_NAME.search(text)
            if redefined:
                found.append((lineno, redefined.group(1), REDEFINITION))

    import io

    with _quiet():
        check(source, filename, _Collector(io.StringIO(), io.StringIO()))
    return sorted(set(found))


def _reported_at(line: int, name: str, reports: Sequence[Tuple[int, str, str]]) -> Optional[str]:
    """名字是不是就在这一行被报成 unused / redefinition。"""

    for lineno, reported, kind in reports:
        if kind == UNUSED and lineno == line and name in _components(reported):
            return UNUSED
    for _, reported, kind in reports:
        if kind == REDEFINITION and name in _components(reported):
            # 重复导入：pyflakes 报的是第二条，人类删的往往是第一条，行号对不上，只按名字认
            return REDEFINITION
    return None


def _components(reported: str) -> Set[str]:
    """``'typing.Dict'`` / ``'os.path'`` / ``'.helpers'`` 里可能对应绑定名的那几段。"""

    cleaned = reported.lstrip(".")
    parts = [part for part in cleaned.split(".") if part]
    if not parts:
        return set()
    return {parts[0], parts[-1]}


def dunder_all_names(source: str, filename: str = "<unknown>") -> Set[str]:
    """模块级 ``__all__`` 里的字符串名字（``=`` 与 ``+=`` 都算）。

    pyflakes 自己会把**字面量** ``__all__`` 里的名字当成「被用到」，所以这一条通常不会
    漏；它只在 ``__all__`` 是拼出来的（``__all__ = BASE + EXTRA``、``__all__.append(...)``）
    时才有用 —— 那种情况下 pyflakes 看不见，删掉就真的改了包的公开 API。
    """

    tree = _parse_quiet(source, filename)
    if tree is None:
        return set()

    names: Set[str] = set()
    for node in ast.walk(tree):
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if value is None:
            continue
        for element in ast.walk(value):
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.add(element.value)
    return names


def has_star_import(source: str, filename: str = "<unknown>") -> bool:
    tree = _parse_quiet(source, filename)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            return True
    return False


def uses_dynamic_lookup(source: str, filename: str = "<unknown>") -> bool:
    """有没有 ``eval`` / ``getattr`` / ``globals()`` 这类把名字藏进字符串的写法。"""

    tree = _parse_quiet(source, filename)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in DYNAMIC_CALLS:
                return True
    return False


def _weaknesses(
    relative: str,
    parent_source: str,
    names: Sequence[str],
    statement: str,
    kind: str,
    exports: Set[str],
) -> List[str]:
    weak: List[str] = []
    if relative.endswith("__init__.py"):
        # 删掉 __init__ 里的 import 等于改包的公开 API，不是「清理未使用导入」
        weak.append(WEAK_PACKAGE_INIT)
    if any(name in exports for name in names):
        weak.append(WEAK_DUNDER_ALL)
    if has_star_import(parent_source, relative):
        weak.append(WEAK_STAR_IMPORT)
    if uses_dynamic_lookup(parent_source, relative):
        weak.append(WEAK_DYNAMIC_USAGE)
    if _is_dotted_side_effect(statement):
        weak.append(WEAK_DOTTED_SIDE_EFFECT)
    if kind == REDEFINITION:
        weak.append(WEAK_REDEFINITION)
    return weak


def _is_dotted_side_effect(statement: str) -> bool:
    """``import a.b``（没有 as）：绑定的是 ``a``，但删掉它会顺带取消 ``a.b`` 的导入副作用。"""

    tree = _parse_quiet(statement)
    if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Import):
        return False
    return any(alias.asname is None and "." in alias.name for alias in tree.body[0].names)


def _removed_imports_and_files(diff: str) -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """从 ``--unified=0`` 的补丁里取「被删掉的 import 行」（含父状态行号）与改动的 py 文件。"""

    imports: List[Tuple[str, str, int]] = []
    files: List[str] = []
    current = ""
    old_line = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[len("+++ b/") :].strip()
            if current.endswith(PY_SUFFIX) and current not in files:
                files.append(current)
            continue
        if raw.startswith("--- "):
            continue
        if raw.startswith("@@"):
            match = re.search(r"-(\d+)", raw)
            old_line = int(match.group(1)) if match else 0
            continue
        if raw.startswith("-"):
            body = raw[1:].strip()
            if IMPORT_LINE.match(body) and not STAR_IMPORT.match(body):
                imports.append((body, current, old_line))
            old_line += 1
        elif raw.startswith("+"):
            # 新增行不占父状态的行号
            continue
        else:
            old_line += 1
    return imports, files


def _diff_removes_import(patch: str) -> bool:
    for raw in patch.splitlines():
        if not raw.startswith("-") or raw.startswith("---"):
            continue
        body = raw[1:].strip()
        if IMPORT_LINE.match(body) and not STAR_IMPORT.match(body):
            return True
    return False


def _cases_in_commit(repo, sha: str, date: str, subject: str, diff: str) -> List[RemovedImportCase]:
    removed, files = _removed_imports_and_files(diff)
    if not removed or not files:
        return []

    parent = repo.run("rev-parse", f"{sha}^", check=False)
    if parent.returncode != 0:
        return []

    cases: List[RemovedImportCase] = []
    for relative in files:
        parent_source = repo.file_at(f"{sha}^", relative)
        if parent_source is None:
            continue
        reports = unused_import_names(parent_source, relative)
        if not reports:
            continue
        exports = dunder_all_names(parent_source, relative)
        for statement, statement_file, statement_line in removed:
            if statement_file and statement_file != relative:
                # import 删在另一个文件里 —— 不是这条缺陷的修法
                continue
            unused: List[str] = []
            kinds: List[str] = []
            for name in import_bindings(statement):
                kind = _reported_at(statement_line, name, reports)
                if kind is None:
                    continue
                unused.append(name)
                kinds.append(kind)
            if not unused:
                continue
            kind = REDEFINITION if REDEFINITION in kinds else UNUSED
            cases.append(
                RemovedImportCase(
                    sha=sha,
                    parent=parent.stdout.strip(),
                    subject=subject,
                    date=date,
                    file=relative,
                    line=statement_line,
                    names=sorted(unused),
                    statement=statement,
                    weak=_weaknesses(relative, parent_source, unused, statement, kind, exports),
                )
            )
    return cases


def mine_repository(
    repo_path: str,
    *,
    limit: Optional[int] = None,
    max_cases: int = 400,
    git=None,
) -> UnusedImportReport:
    """扫描最近 ``limit`` 个提交（``None`` = 整个历史），返回所有「未使用 import ← 人类删了它」的配对。"""

    from .import_fix_miner import iter_commit_patches
    from .real_defect_benchmark import GitRepo

    repo = git or GitRepo(repo_path)
    report = UnusedImportReport(repo=repo_path)
    failures: List[str] = []
    total = limit if limit is not None else repo.count_commits()

    for sha, date, subject, patch in iter_commit_patches(repo, total, failures=failures):
        report.scanned_commits += 1
        if len(report.cases) >= max_cases:
            break
        if not _diff_removes_import(patch):
            continue
        report.candidate_commits += 1
        report.cases.extend(_cases_in_commit(repo, sha, date, subject, patch))

    report.notes.extend(failures)
    if not report.cases:
        report.notes.append("没有找到配对：可能是历史太短、该项目从未清理过未使用导入，或它在引入 lint 之前就已干净")
    return report
