"""从一个真实仓库的历史里挖出「父状态有未定义名 → 子提交补上了 import」的配对。

这是给 ``undefined_name`` 这一类缺陷准备的题库。与语义缺陷不同，这类缺陷的判据可以
做到完全干净，因为有**两个互相独立的事实**：

* 父状态真的缺一个名字 —— 由 pyflakes 判定（第三方工具，不是我们自己的感知器，
  否则「检出率」就是自己给自己打分）；
* 人类当年补了什么 —— 由提交里**新增**的 import 行判定。

有了这两条，「系统补的」才能与「人类补的」对齐比较：来源一致率、放弃率、假阳性率
都建立在它们之上。

五个已知会污染样本的陷阱，都在这里显式标记（而不是默默收下）：

1. 只认**新增行**——同一个提交里删掉未使用 import 与新增 import 混在一起时，
   只看新增，否则会把「人类删了 import」误当成「人类加的 import」；
2. ``if TYPE_CHECKING:`` 块里的 import —— 父状态报 F821，但运行时本来不触发；
3. 注释 / 文档字符串里的名字 —— pyflakes 不会报（F821 只在语法树里有意义），
   天然被排除；
4. 父状态那一行带 ``# noqa`` —— 人类选择压制而不是修复，这是「拒绝修」的对照样本；
5. 跨文件——F821 在 A 文件而 import 加在 B 文件（常见于 ``__init__`` 再导出）。

标记为 ``weak`` 的用例不进主指标，但会被报告出来，因为它们正是「放弃率」的天然样本。
"""

from __future__ import annotations

import ast
import contextlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

IMPORT_LINE = re.compile(r"^(import\s+[\w.]+|from\s+[\w.]+\s+import\s+.+)$")
UNDEFINED_NAME = re.compile(r"undefined name '([^']+)'")
NOQA = re.compile(r"#\s*noqa\b", re.IGNORECASE)
PY_SUFFIX = ".py"

WEAK_TYPE_CHECKING = "type_checking"
WEAK_CROSS_FILE = "cross_file"
WEAK_SUPPRESSED = "human_suppressed_with_noqa"
WEAK_PY2 = "python2_builtin"
WEAK_TYPING_ONLY = "typing_only"
WEAK_NOT_RESOLVED = "import_does_not_resolve"

# Python 2 里是内建、Python 3 里不存在的名字。父状态在这一批名字上报 F821，说明那段
# 代码是 py2 遗留分支，人类当年的修法是**删掉分支**而不是补 import —— 它们不是活的缺陷，
# 也永远不会和「新增 import」配上（实测 requests 最近 2500 个提交里 82 个 F821 全是它们）。
PY2_ONLY_BUILTINS = frozenset(
    {
        "unicode",
        "long",
        "basestring",
        "unichr",
        "xrange",
        "raw_input",
        "execfile",
        "buffer",
        "coerce",
        "intern",
        "apply",
        "cmp",
        "file",
        "reduce",
        "StandardError",
    }
)

# 只在类型检查器里存在的名字。运行时本来就不该有绑定（``reveal_type`` 之类），
# 父状态报 F821 是工具差异，不是缺陷（实测 attrs 最近 1800 个提交里 23 个 F821 全是它们）。
TYPING_ONLY_NAMES = frozenset(
    {
        "reveal_type",
        "reveal_locals",
        "assert_type",
        "TYPE_CHECKING",
        "TypeGuard",
        "TypeIs",
        "Never",
        "NoReturn",
        "LiteralString",
        "TypeAlias",
        "dataclass_transform",
    }
)


@dataclass
class ImportFixCase:
    """一条可对齐的样本：父状态缺 ``name``，人类在 ``sha`` 里补上了 import。"""

    sha: str
    parent: str
    subject: str
    date: str
    file: str
    name: str
    human_statement: str
    human_key: Tuple[str, str]
    human_line: int
    resolved: bool = True
    use_sites: int = 1
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
            "name": self.name,
            "human_statement": self.human_statement,
            "human_key": list(self.human_key),
            "human_line": self.human_line,
            "resolved": self.resolved,
            "use_sites": self.use_sites,
            "weak": self.weak,
            "clean": self.is_clean,
        }


@dataclass
class MineReport:
    repo: str = ""
    scanned_commits: int = 0
    candidate_commits: int = 0
    cases: List[ImportFixCase] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def clean(self) -> List[ImportFixCase]:
        return [case for case in self.cases if case.is_clean]

    @property
    def weak_reasons(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for case in self.cases:
            for reason in case.weak:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def summary_line(self) -> str:
        return (
            f"{self.repo}：扫描 {self.scanned_commits} 个提交，其中 {self.candidate_commits} 个新增了 import；"
            f"得到 {len(self.cases)} 条配对（干净 {len(self.clean)} 条）"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo": self.repo,
            "scanned_commits": self.scanned_commits,
            "candidate_commits": self.candidate_commits,
            "pairs": len(self.cases),
            "clean_pairs": len(self.clean),
            "weak_reasons": self.weak_reasons,
            "cases": [case.to_dict() for case in self.cases],
            "notes": self.notes,
        }


def undefined_names(source: str, filename: str) -> List[Tuple[int, str]]:
    """用 pyflakes 找出未定义名，返回 ``[(行号, 名字)]``（第三方判据，不是自己的感知器）。"""

    check, Reporter = _load_grader()
    found: List[Tuple[int, str]] = []

    class _Collector(Reporter):
        def unexpectedError(self, path, message):  # noqa: N802 - pyflakes 接口
            pass

        def syntaxError(self, path, message, lineno, offset, text):  # noqa: N802
            pass

        def flake(self, message):
            match = UNDEFINED_NAME.search(str(message))
            if match:
                found.append((int(getattr(message, "lineno", 0) or 0), match.group(1)))

    import io

    with _quiet():
        check(source, filename, _Collector(io.StringIO(), io.StringIO()))
    return sorted(set(found))


def _load_grader():
    """取第三方判据（pyflakes）。缺了它**必须炸**，不能悄悄返回空。

    悄悄返回空会被读成「这个项目从没修过未定义名」—— 一个由环境缺失编出来的结论，
    正是本项目最该避免的那类假报告。
    """

    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError as error:
        raise GraderUnavailable(
            "挖未定义名需要 pyflakes 当第三方判据（flake8 会把它一起装上）；"
            "装了才能挖，因为「判不出来」和「没有缺陷」是两件事。"
        ) from error
    return check, Reporter


class GraderUnavailable(RuntimeError):
    """缺少第三方判据。与「没有找到缺陷」严格区分。"""


@contextlib.contextmanager
def _quiet():
    """吞掉旧代码在新解释器下刷屏的 ``SyntaxWarning``：那是环境差异，不是判据。"""

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _parse_quiet(source: str, filename: str = "<unknown>") -> Optional[ast.Module]:
    """``ast.parse``，但走上面的静音通道（``ast.parse`` 本身也会报 SyntaxWarning）。"""

    with _quiet():
        try:
            return ast.parse(source, filename=filename)
        except (SyntaxError, ValueError):
            return None


def import_bindings(statement: str) -> Dict[str, Tuple[str, str]]:
    """一条 import 语句绑定了哪些名字 -> ``名字: (来源模块, 原始名字)``。

    ``import os`` -> ``{"os": ("", "os")}``（来源模块为空表示「整个模块」）；
    ``from typing import List`` -> ``{"List": ("typing", "List")}``。
    """

    tree = _parse_quiet(statement)
    if tree is None:
        return {}
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
        return {}

    node = tree.body[0]
    bindings: Dict[str, Tuple[str, str]] = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            bindings[alias.asname or alias.name.split(".")[0]] = ("", alias.name)
    else:
        module = node.module or ""
        for alias in node.names:
            bindings[alias.asname or alias.name] = (module, alias.name)
    return bindings


def mine_repository(
    repo_path: str,
    *,
    limit: Optional[int] = None,
    max_cases: int = 200,
    git=None,
) -> MineReport:
    """扫描最近 ``limit`` 个提交（``None`` = 整个历史），返回所有配对。"""

    from .real_defect_benchmark import GitRepo

    repo = git or GitRepo(repo_path)
    report = MineReport(repo=repo_path)
    failures: List[str] = []
    total = limit if limit is not None else repo.count_commits()

    for sha, date, subject, patch in iter_commit_patches(repo, total, failures=failures):
        report.scanned_commits += 1
        if len(report.cases) >= max_cases:
            break
        if not _diff_adds_import(patch):
            continue
        report.candidate_commits += 1
        report.cases.extend(_cases_in_commit(repo, sha, date, subject, patch))

    report.notes.extend(failures)

    if not report.cases:
        report.notes.append("没有找到干净的配对：可能是历史太短、该项目从未修过未定义名，或它在引入 lint 之前就已干净")
    return report


def iter_commit_patches(repo, total: int, *, chunk: int = 200, failures: Optional[List[str]] = None):
    """分块产出 ``(sha, date, subject, patch)``，内存里一次只留一块。

    一次 ``git log -p`` 拿几千个提交的完整补丁，会把 git for Windows 的信号管道撑爆
    （cygwin ``sh.exe`` 报 ``Win32 error 5``），所以按块取、按块丢。某一块失败就减半重试，
    减到 1 个提交还失败就如实记一笔，而不是假装扫描过。
    """

    skip = 0
    while skip < total:
        size = min(chunk, total - skip)
        while True:
            try:
                records = repo.log_with_patches(size, skip=skip)
                break
            except Exception as error:  # noqa: BLE001 - git 在受限 shell 下会以各种方式失败
                if size <= 1:
                    if failures is not None:
                        failures.append(f"跳过第 {skip + 1} 个提交：{error}")
                    records = []
                    break
                size = max(1, size // 2)
        yield from records
        skip += size


def _diff_adds_import(patch: str) -> bool:
    for raw in patch.splitlines():
        if raw.startswith("+") and not raw.startswith("+++") and IMPORT_LINE.match(raw[1:].strip()):
            return True
    return False


def _cases_in_commit(repo, sha: str, date: str, subject: str, diff: str) -> List[ImportFixCase]:
    added, files = _added_imports_and_files(diff)
    if not added or not files:
        return []

    parent = repo.run("rev-parse", f"{sha}^", check=False)
    if parent.returncode != 0:
        return []

    cases: List[ImportFixCase] = []
    child_sources: Dict[str, Optional[str]] = {}
    for relative in files:
        parent_source = repo.file_at(f"{sha}^", relative)
        if parent_source is None:
            continue
        missing = undefined_names(parent_source, relative)
        if not missing:
            continue
        # 同一个名字在一个文件里可能被引用多次（pyflakes 每个使用点各报一次）。一条「缺陷」
        # 是 (提交, 文件, 名字)，不是每个使用点 —— 否则 1 条样本会被放大成 N 条。
        use_sites: Dict[str, int] = {}
        first_line: Dict[str, int] = {}
        for lineno, name in missing:
            use_sites[name] = use_sites.get(name, 0) + 1
            first_line.setdefault(name, lineno)
        child_source = repo.file_at(sha, relative)
        child_missing = {name for _, name in (undefined_names(child_source, relative) if child_source else [])}
        for name, sites in use_sites.items():
            lineno = first_line[name]
            for statement, statement_file, statement_line in added:
                key = import_bindings(statement).get(name)
                if key is None:
                    continue
                if statement_file not in child_sources:
                    child_sources[statement_file] = repo.file_at(sha, statement_file) if statement_file else None
                cases.append(
                    ImportFixCase(
                        sha=sha,
                        parent=parent.stdout.strip(),
                        subject=subject,
                        date=date,
                        file=relative,
                        name=name,
                        human_statement=statement,
                        human_key=key,
                        human_line=statement_line,
                        resolved=name not in child_missing,
                        use_sites=sites,
                        weak=_weaknesses(
                            repo,
                            sha,
                            relative,
                            lineno,
                            parent_source,
                            statement_file,
                            statement_line,
                            name,
                            child_sources[statement_file],
                            child_missing,
                        ),
                    )
                )
    return cases


def _added_imports_and_files(diff: str) -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """从 ``git show --unified=0`` 的输出里取出「新增的 import 行」与「改动的 py 文件」。"""

    imports: List[Tuple[str, str, int]] = []
    files: List[str] = []
    current = ""
    new_line = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[len("+++ b/") :].strip()
            if current.endswith(PY_SUFFIX) and current not in files:
                files.append(current)
            continue
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            new_line = int(match.group(1)) if match else 0
            continue
        if raw.startswith("+"):
            body = raw[1:].strip()
            if IMPORT_LINE.match(body):
                imports.append((body, current, new_line))
            new_line += 1
    return imports, files


def _weaknesses(
    repo,
    sha: str,
    relative: str,
    lineno: int,
    parent_source: str,
    statement_file: str,
    statement_line: int,
    name: str,
    child_source: Optional[str],
    child_missing: Set[str],
) -> List[str]:
    weak: List[str] = []
    if name in PY2_ONLY_BUILTINS:
        # py2 遗留分支：人类当年的修法是删掉分支，不是补 import
        weak.append(WEAK_PY2)

    if name in TYPING_ONLY_NAMES:
        # 只在类型检查器里存在的名字：运行时本来就没有绑定
        weak.append(WEAK_TYPING_ONLY)

    if statement_file and statement_file != relative:
        # F821 在 A 文件、import 加在 B 文件：oracle 大概率放弃，这是能力边界不是错误
        weak.append(WEAK_CROSS_FILE)

    if _line_has_noqa(parent_source, lineno):
        weak.append(WEAK_SUPPRESSED)

    # 人类把 import 加在 `if TYPE_CHECKING:` 里：父状态报 F821 但运行时本来不触发
    if child_source and _line_is_inside_type_checking(child_source, statement_file, statement_line):
        weak.append(WEAK_TYPE_CHECKING)

    if name in child_missing:
        # 加了 import 之后这个名字在那个文件里仍然未定义 —— 说明这条 import 没修它，
        # 配对不成立（这是「配上」的硬条件，不是加分项）
        weak.append(WEAK_NOT_RESOLVED)
    return weak


def _line_has_noqa(source: str, lineno: int) -> bool:
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return bool(NOQA.search(lines[lineno - 1]))
    return False


def _line_is_inside_type_checking(source: str, filename: str, lineno: int) -> bool:
    return any(start <= lineno <= end for start, end in _type_checking_ranges(source, filename))


def _type_checking_ranges(source: str, filename: str) -> List[Tuple[int, int]]:
    """``if TYPE_CHECKING:`` 块的 (起始行, 结束行)。"""

    tree = _parse_quiet(source, filename)
    if tree is None:
        return []

    ranges: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            start = node.lineno
            end = max((child.end_lineno or child.lineno) for child in node.body)
            ranges.append((start, end))
    return ranges


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def candidate_commits(repo_path: str, limit: int) -> int:
    """只做计数：最近 ``limit`` 个提交里有多少个新增了 import。"""

    from .real_defect_benchmark import GitRepo

    return len(GitRepo(repo_path).log_adding_imports(limit))
