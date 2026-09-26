"""成功声明审计：把「自称成功」和「证成成功」分开。

论文里每一个「成功」都应该来自一组判据，而不是一句断言。这个模块用 AST 静态扫描
成功声明的调用点（``print("... SUCCESS ...")``、``report_success()``、``return "SUCCESS"``），
数出这条声明真正被几条判据守着；少于 ``MIN_CRITERIA`` 条的一律标成 ``UNVERIFIED_CLAIM``，
再由 :class:`~hermes.self_research.integrity_checker.IntegrityChecker` 强制写进论文的免责声明。

取向与 Perceiver 一致：**宁可漏报，不可误报**。

认领的位置，都是「产出 / 播报」，不是「读值」：

  * 播报调用（print / log / report / emit / ...）的实参里出现 ``SUCCESS`` 文本；
  * ``report_success()`` 这类以「声明成功」为动作的函数调用；
  * 返回值恰好等于 ``SUCCESS`` 的 ``return``（``return "SUCCESS"``、``return "SUCCESS", []``）。

不认领：

  * ``x == "SUCCESS"``——这是比较，判据在别处；
  * ``"SUCCESS" if ok else "FAILED"``——这是渲染一个已经算出来的布尔量；
  * ``{"status": "SUCCESS"}``——嵌在字典里的样本数据，不是结论。

判据怎么数（``MIN_CRITERIA`` = 3）：

  * 声明被 ``if`` 包着：数包住它的那些 ``if``（``and`` 的每个操作数各算一条）；
  * 声明是函数体里的裸语句：往回数**紧邻**的「拒绝式守卫」——函数体里直接
    ``return``/``raise``/``continue``/``break`` 的 ``if``，或向「拒绝累加器」
    （先 append、后面又被判断或返回的那个名字）追加理由的 ``if``。

判据必须紧邻、且真正参与决策，就是为了不把长函数里一串无关的 ``if`` 算成判据。
"""

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

MIN_CRITERIA = 3

VERDICT_VERIFIED = "VERIFIED"
VERDICT_UNVERIFIED = "UNVERIFIED_CLAIM"

# 「恰好等于这些词」才算结论；长句子里出现 SUCCESS 只算播报，不算判定。
VERDICT_TOKENS = ("SUCCESS", "SUCCESSFUL", "SUCCEEDED", "PASS", "PASSED", "OK")

SUCCESS_TEXT_RE = re.compile(r"\bSUCCESS\b")
FAILURE_TEXT_RE = re.compile(r"\b(FAIL|FAILED|FAILURE|INCONCLUSIVE|REJECT|REJECTED|ERROR)\b")
SUCCESS_CALL_RE = re.compile(r"^_?(report|record|mark|declare|claim|log)_success(ful|_?claim)?$")
EMITTER_RE = re.compile(
    r"^(print|pprint|echo|say|report\w*|emit\w*|record\w*|mark\w*|write\w*|audit\w*|"
    r"warn\w*|log\w*|logger|debug|info|warning|error|critical|exception)$"
)

ACCUMULATOR_METHODS = ("append", "add", "update", "setdefault", "extend")

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_BLOCK_ATTRS = ("body", "orelse", "finalbody")

# 这些目录里不放本仓库的源码：要么是依赖、要么是缓存、要么是别人的测试。
DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "site-packages",
        ".tmp_test",
        ".refactor_backups",
        ".workbuddy",
        ".uploads",
        ".idea",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "build",
        "dist",
        "tests",
    }
)


@dataclass(frozen=True)
class ClaimCriterion:
    """一条「判据」：一个必须成立的守卫条件。"""

    line: int
    kind: str
    test: str

    def to_dict(self) -> Dict[str, Any]:
        return {"line": self.line, "kind": self.kind, "test": self.test}


@dataclass(frozen=True)
class SuccessClaim:
    """一条成功声明，以及它到底被几条判据守着。"""

    file: str
    line: int
    function: str
    kind: str
    statement: str
    criteria: Tuple[ClaimCriterion, ...]
    verdict: str
    reasons: Tuple[str, ...]

    @property
    def is_verified(self) -> bool:
        return self.verdict == VERDICT_VERIFIED

    @property
    def claim_id(self) -> str:
        return f"claim:{self.file}:{self.line}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "file": self.file,
            "line": self.line,
            "function": self.function,
            "kind": self.kind,
            "statement": self.statement,
            "criteria": [item.to_dict() for item in self.criteria],
            "criteria_count": len(self.criteria),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


@dataclass
class ClaimAuditReport:
    """一次成功声明审计的结果。"""

    claims: List[SuccessClaim] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)
    scanned_files: int = 0
    skipped_roots: List[str] = field(default_factory=list)
    unparsable: List[str] = field(default_factory=list)
    min_criteria: int = MIN_CRITERIA
    note: str = ""

    @property
    def unverified(self) -> List[SuccessClaim]:
        return [claim for claim in self.claims if not claim.is_verified]

    @property
    def verified_count(self) -> int:
        return len(self.claims) - len(self.unverified)

    @property
    def is_clean(self) -> bool:
        return not self.unverified

    def summary_line(self) -> str:
        if not self.claims:
            return f"成功声明审计通过：扫描 {self.scanned_files} 个源文件，" f"未发现成功声明调用点（未做任何推断）。"
        if self.is_clean:
            return f"成功声明审计通过：{len(self.claims)} 处成功声明全部有 " f">= {self.min_criteria} 条判据。"
        return (
            f"成功声明审计：{len(self.claims)} 处成功声明里 {len(self.unverified)} 处未经证成"
            f"（判据少于 {self.min_criteria} 条）。"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "summary": self.summary_line(),
            "roots": list(self.roots),
            "scanned_files": self.scanned_files,
            "skipped_roots": list(self.skipped_roots),
            "unparsable": list(self.unparsable),
            "min_criteria": self.min_criteria,
            "note": self.note,
            "claims": [claim.to_dict() for claim in self.claims],
            "unverified_claims": [claim.claim_id for claim in self.unverified],
        }


class SuccessClaimAuditor:
    """静态扫描成功声明，判断它是否被足够的判据守着。"""

    def __init__(
        self,
        roots: Optional[Sequence[str]] = None,
        min_criteria: int = MIN_CRITERIA,
        skip_dirs: Iterable[str] = DEFAULT_SKIP_DIRS,
    ):
        self._roots = [str(item) for item in roots] if roots else None
        self._min_criteria = int(min_criteria)
        self._skip_dirs = frozenset(skip_dirs)

    # ------------------------------------------------------------ 对外接口

    def audit_paths(self, roots: Optional[Sequence[str]] = None) -> ClaimAuditReport:
        targets = [str(item) for item in (roots or self._roots or [discover_repo_root()])]
        report = ClaimAuditReport(roots=targets, min_criteria=self._min_criteria)
        missing = False
        for target in targets:
            path = Path(target)
            if not path.exists():
                report.skipped_roots.append(target)
                missing = True
                continue
            for file_path in self._iter_files(path):
                report.scanned_files += 1
                try:
                    source = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # 便宜的子串预筛：省掉对绝大多数文件的 ast.parse。
                if "SUCCESS" not in source and "_success" not in source:
                    continue
                name = _display_name(file_path)
                try:
                    ast.parse(source, filename=name)
                except SyntaxError:
                    report.unparsable.append(name)
                    continue
                report.claims.extend(self.audit_source(source, name))
        if missing and not report.claims:
            report.note = "扫描根目录不存在，本次没有可审计的源码（未做任何推断）。"
        report.claims.sort(key=lambda claim: (claim.file, claim.line, claim.function))
        return report

    def audit_source(self, source: str, filename: str = "<source>") -> List[SuccessClaim]:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return []

        parents = _build_parents(tree)
        excluded = _excluded_literal_ids(tree)
        claims: List[SuccessClaim] = []
        for node, kind in _claim_sites(tree, excluded):
            function = _enclosing_function(parents, node)
            criteria = self._criteria_for(parents, function, node)
            reasons: List[str] = []
            if len(criteria) < self._min_criteria:
                detail = "、".join(f"L{item.line} {item.test}" for item in criteria) or "没有任何守卫条件"
                reasons.append(f"只有 {len(criteria)} 条判据（要求 >= {self._min_criteria}）：{detail}")
            claims.append(
                SuccessClaim(
                    file=filename,
                    line=int(getattr(node, "lineno", 0)),
                    function=function.name if function is not None else "<module>",
                    kind=kind,
                    statement=_statement_text(node),
                    criteria=tuple(criteria),
                    verdict=VERDICT_VERIFIED if not reasons else VERDICT_UNVERIFIED,
                    reasons=tuple(reasons),
                )
            )
        claims.sort(key=lambda claim: (claim.line, claim.function))
        return claims

    # ------------------------------------------------------------ 内部实现

    def _iter_files(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            if root.suffix == ".py":
                yield root
            return
        for current, dirs, names in os.walk(str(root), topdown=True):
            dirs[:] = sorted(item for item in dirs if item not in self._skip_dirs)
            for name in sorted(names):
                if name.endswith(".py"):
                    yield Path(current) / name

    def _criteria_for(
        self,
        parents: Dict[ast.AST, ast.AST],
        function: Optional[ast.AST],
        claim: ast.AST,
    ) -> List[ClaimCriterion]:
        if function is None:
            # 模块级的裸声明：没有任何守卫，直接算 0 条判据。
            return []

        path = _path_between(parents, function, claim)
        enclosing = [node for node in path if isinstance(node, ast.If)]
        if enclosing:
            criteria: List[ClaimCriterion] = []
            for node in enclosing:
                criteria.extend(_criteria_of(node, "enclosing"))
            return criteria

        block = _containing_block(parents.get(claim), claim)
        if block is None:
            return []
        index = next((position for position, item in enumerate(block) if item is claim), None)
        if index is None:
            return []

        accumulators = _rejection_accumulators(function)
        guards: List[ast.If] = []
        for statement in reversed(block[:index]):
            if not isinstance(statement, ast.If) or not _is_rejection_guard(statement, accumulators):
                break
            guards.append(statement)
        guards.reverse()

        criteria = []
        for node in guards:
            criteria.extend(_criteria_of(node, "rejection_guard"))
        return criteria


# ------------------------------------------------------------------ 模块级工具


def discover_repo_root(start: Optional[str] = None) -> str:
    """找出要审计的仓库根目录：优先调用方给的路径，其次 cwd，最后模块自身的位置。"""

    module_root = Path(__file__).resolve()
    candidates: List[Optional[str]] = [start, os.getcwd()]
    if len(module_root.parents) > 3:
        candidates.append(str(module_root.parents[3]))

    fallback = os.getcwd()
    for item in candidates:
        if not item:
            continue
        path = Path(item).resolve()
        if not path.exists():
            continue
        fallback = str(path)
        for candidate in [path] + list(path.parents):
            if (candidate / "autotestgen.py").is_file() and (candidate / "src").is_dir():
                return str(candidate)
    return fallback


def _display_name(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path(discover_repo_root()))
    except (ValueError, OSError):
        relative = path
    return str(relative).replace("\\", "/")


def _callee_name(call: ast.Call) -> Optional[str]:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _build_parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents.setdefault(child, node)
    return parents


def _excluded_literal_ids(tree: ast.AST) -> Set[int]:
    """比较操作数与三元表达式里的常量：那是读值 / 渲染，不是下结论。"""

    excluded: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Compare, ast.IfExp)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    excluded.add(id(sub))
    return excluded


def _iter_scope(node: ast.AST) -> Iterator[ast.AST]:
    """遍历 node 的子孙，但不进入嵌套函数 / 类的作用域。

    ``node`` 自己不算作用域边界——所以 ``_iter_scope(function_def)`` 拿到的是这个
    函数自己的函数体，而不是空集。
    """

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, _NESTED_SCOPES):
            stack.extend(ast.iter_child_nodes(child))


def _body_nodes(if_node: ast.If) -> Iterator[ast.AST]:
    for statement in if_node.body:
        yield statement
        if not isinstance(statement, _NESTED_SCOPES):
            yield from _iter_scope(statement)


def _path_between(parents: Dict[ast.AST, ast.AST], root: ast.AST, node: ast.AST) -> List[ast.AST]:
    path: List[ast.AST] = []
    current: Optional[ast.AST] = node
    while current is not None and current is not root:
        path.append(current)
        current = parents.get(current)
    return path


def _enclosing_function(parents: Dict[ast.AST, ast.AST], node: ast.AST) -> Optional[ast.AST]:
    current: Optional[ast.AST] = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _containing_block(parent: Optional[ast.AST], node: ast.AST) -> Optional[List[ast.AST]]:
    if parent is None:
        return None
    for attr in _BLOCK_ATTRS:
        block = getattr(parent, attr, None)
        if isinstance(block, list) and any(item is node for item in block):
            return block
    return None


def _criteria_of(if_node: ast.If, kind: str) -> List[ClaimCriterion]:
    test = if_node.test
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        conjuncts = list(test.values)
    else:
        conjuncts = [test]
    return [ClaimCriterion(line=int(if_node.lineno), kind=kind, test=_safe_unparse(item)) for item in conjuncts]


def _rejection_accumulators(function: ast.AST) -> Set[str]:
    """被 append 过、并且后面真的参与了判断或返回的名字（比如 ``reasons``）。"""

    appended: Set[str] = set()
    for node in _iter_scope(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ACCUMULATOR_METHODS
        ):
            base = node.func.value
            if isinstance(base, ast.Name):
                appended.add(base.id)
    if not appended:
        return set()

    consulted: Set[str] = set()
    for node in _iter_scope(function):
        if isinstance(node, ast.If):
            consulted.update(item.id for item in ast.walk(node.test) if isinstance(item, ast.Name))
        elif isinstance(node, ast.Return) and node.value is not None:
            consulted.update(item.id for item in ast.walk(node.value) if isinstance(item, ast.Name))
    return appended & consulted


def _is_rejection_guard(if_node: ast.If, accumulators: Set[str]) -> bool:
    """这个 ``if`` 是「不满足就不算成功」的守卫吗？"""

    for node in _body_nodes(if_node):
        if isinstance(node, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            value = getattr(node, "value", None)
            if value is None or not _is_success_verdict(value):
                return True
        elif isinstance(node, ast.Call):
            name = _callee_name(node)
            if name and EMITTER_RE.match(name) and _mentions_failure(node):
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ACCUMULATOR_METHODS:
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in accumulators:
                    return True
    return False


def _mentions_success(call: ast.Call, excluded: Set[int]) -> bool:
    for value in list(call.args) + [keyword.value for keyword in call.keywords]:
        for sub in ast.walk(value):
            if (
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and SUCCESS_TEXT_RE.search(sub.value)
                and id(sub) not in excluded
            ):
                return True
    return False


def _mentions_failure(call: ast.Call) -> bool:
    for value in list(call.args) + [keyword.value for keyword in call.keywords]:
        for sub in ast.walk(value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and FAILURE_TEXT_RE.search(sub.value):
                return True
    return False


def _is_verdict_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() in VERDICT_TOKENS


def _is_empty_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in (None, "", 0)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    return False


def _is_success_verdict(value: ast.AST) -> bool:
    """返回值本身就是一个「成功」结论，而不是一段数据或一句话。"""

    if isinstance(value, ast.Constant):
        return _is_verdict_token(value.value)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        if not value.elts:
            return False
        return all(_is_success_verdict(item) or _is_empty_literal(item) for item in value.elts)
    return False


def _claim_sites(tree: ast.AST, excluded: Set[int]) -> List[Tuple[ast.AST, str]]:
    sites: Dict[Tuple[int, int], Tuple[ast.AST, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = _callee_name(call)
            if name and EMITTER_RE.match(name) and _mentions_success(call, excluded):
                sites.setdefault((call.lineno, call.col_offset), (call, "emitted_claim"))
        if isinstance(node, ast.Call):
            name = _callee_name(node)
            if name and SUCCESS_CALL_RE.match(name):
                sites[(node.lineno, node.col_offset)] = (node, "report_call")
        if isinstance(node, ast.Return) and node.value is not None and _is_success_verdict(node.value):
            sites.setdefault((node.lineno, node.col_offset), (node, "verdict_return"))
    return [sites[key] for key in sorted(sites)]


def _safe_unparse(node: ast.AST) -> str:
    try:
        return " ".join(ast.unparse(node).split())
    except Exception:  # pragma: no cover - unparse 对合法 AST 不会失败
        return "<unparsable>"


def _statement_text(node: ast.AST) -> str:
    return _safe_unparse(node)[:160]
