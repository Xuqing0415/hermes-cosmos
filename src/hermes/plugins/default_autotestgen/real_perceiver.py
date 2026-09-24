"""默认（Python）域的真实感知器：基于 AST 的静态检查，报出的位置全部来自真实节点。

设计原则是「宁可漏报，不可误报」——一个编造位置的感知器比一个漏报的感知器危险得多：

1. 只用标准库 ``ast``，不引入 pyflakes / pylint；
2. 未定义名采用「整个模块里绑定过就不报」这种保守判据，不去重算作用域规则，
   从而不会因为作用域算错而误报；
3. 模块里有 ``from x import *`` 时直接放弃未定义名判断（导入内容不可知）；
4. ``__init__.py`` 不检查未使用导入，那正是再导出的常见写法；
5. 语法错误、编码未知的文件直接跳过，不猜。

每个 ``PainPoint.location`` 都是相对扫描根目录的正斜杠路径，``context["line"]``
是真实行号：两者都取自 AST 节点，没有一处写死。
"""

from __future__ import annotations

import ast
import builtins
import os
from typing import Dict, List, Optional, Set, Tuple

from hermes.core.interfaces import DomainContext, PainPoint, Perceiver

SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    ".venv",
    "venv",
    "env",
    "ENV",
    "node_modules",
    "site-packages",
    "build",
    "dist",
    ".tmp_test",
    ".refactor_backups",
    ".workbuddy",
    ".uploads",
}

#: 扫描上限：只影响超大目录树的耗时，不改变报出的位置是真是假。
MAX_FILES = 5000

BUILTIN_NAMES = frozenset(dir(builtins))

#: 由解释器或模块机制隐式提供的名字，不该被当成「从未定义」。
IMPLICIT_GLOBALS = frozenset(
    {
        "__file__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__debug__",
        "__dict__",
        "__path__",
        "__annotations__",
    }
)

SEVERITY_UNDEFINED = "high"
SEVERITY_UNUSED_IMPORT = "low"


class _NameCollector(ast.NodeVisitor):
    """收集模块里被绑定的名字与被读取的名字（不区分作用域，因而偏保守）。"""

    def __init__(self) -> None:
        self.bound: Set[str] = set()
        self.loaded: Dict[str, Tuple[int, int]] = {}
        self.has_star_import = False
        self.string_constants: Set[str] = set()

    # ------------------------------------------------------------------ 读取
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.setdefault(node.id, (node.lineno, node.col_offset))
        else:
            self.bound.add(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.string_constants.add(node.value)
        self.generic_visit(node)

    # ------------------------------------------------------------------ 绑定
    def visit_arg(self, node: ast.arg) -> None:
        self.bound.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        self.bound.add(node.asname or node.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            self.has_star_import = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.bound.update(node.names)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.bound.update(node.names)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.bound.add(node.rest)
        self.generic_visit(node)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _module_imports(tree: ast.Module) -> List[Tuple[str, str, int]]:
    """返回模块顶层的 ``(本地名, 来源描述, 行号)``。

    只认最外层语句与 ``if TYPE_CHECKING:`` 块——放在 ``try/except`` 或版本判断里的
    导入往往是可选依赖，那种情况下「未使用」的判断不可靠，直接不看。
    """

    found: List[Tuple[str, str, int]] = []

    def collect(statements: List[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    found.append((alias.asname or alias.name.split(".")[0], f"import {alias.name}", statement.lineno))
            elif isinstance(statement, ast.ImportFrom):
                module = statement.module or "."
                if module == "__future__":
                    # `from __future__ import annotations` 这类导入本来就不会被当名字使用
                    continue
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    found.append((alias.asname or alias.name, f"from {module} import {alias.name}", statement.lineno))

    collect(tree.body)
    for statement in tree.body:
        if isinstance(statement, ast.If) and _is_type_checking(statement.test):
            collect(statement.body)
    return found


class RealPerceiver(Perceiver):
    """真实静态检查：查不出问题时返回空列表，而不是编造问题来显得有用。"""

    def __init__(self, max_files: int = MAX_FILES):
        self.max_files = max(1, max_files)

    def detect(self, context: DomainContext) -> List[PainPoint]:
        root = os.path.abspath(context.path or ".")
        pain_points: List[PainPoint] = []
        for path in self._python_files(root):
            tree = self._parse(path)
            if tree is None:
                continue
            relative = os.path.relpath(path, root).replace("\\", "/")
            pain_points.extend(self._undefined_names(tree, relative))
            if os.path.basename(path) != "__init__.py":
                pain_points.extend(self._unused_imports(tree, relative))
        return pain_points

    # ------------------------------------------------------------------ 工具
    def _python_files(self, root: str) -> List[str]:
        if os.path.isfile(root):
            return [root] if root.endswith(".py") else []

        found: List[str] = []
        for base, directories, files in os.walk(root):
            directories[:] = sorted(name for name in directories if name not in SKIP_DIRECTORIES)
            for name in sorted(files):
                if name.endswith(".py"):
                    found.append(os.path.join(base, name))
                    if len(found) >= self.max_files:
                        return found
        return found

    @staticmethod
    def _parse(path: str) -> Optional[ast.Module]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            return None
        try:
            return ast.parse(source, filename=path)
        except (SyntaxError, ValueError):
            return None

    # ------------------------------------------------------------------ 检查
    def _undefined_names(self, tree: ast.Module, relative: str) -> List[PainPoint]:
        collector = _NameCollector()
        collector.visit(tree)
        if collector.has_star_import:
            # 有 `from x import *` 时导入内容不可知，「未定义」判断必然不可靠
            return []

        issues: List[PainPoint] = []
        for name, (lineno, column) in sorted(collector.loaded.items(), key=lambda item: item[1]):
            if name in collector.bound or name in BUILTIN_NAMES or name in IMPLICIT_GLOBALS:
                continue
            issues.append(
                self._pain_point(
                    relative,
                    lineno,
                    column,
                    "undefined_name",
                    SEVERITY_UNDEFINED,
                    f"使用了从未定义的名字 {name!r}",
                )
            )
        return issues

    def _unused_imports(self, tree: ast.Module, relative: str) -> List[PainPoint]:
        imports = _module_imports(tree)
        if not imports:
            return []

        collector = _NameCollector()
        collector.visit(tree)
        # 字符串常量也算「用到」：``__all__`` 与字符串注解都写在字符串里
        used = set(collector.loaded) | collector.string_constants

        issues: List[PainPoint] = []
        seen: Set[str] = set()
        for name, source, lineno in imports:
            if name in used or name in seen:
                continue
            seen.add(name)
            issues.append(
                self._pain_point(
                    relative,
                    lineno,
                    0,
                    "unused_import",
                    SEVERITY_UNUSED_IMPORT,
                    f"导入了 {source}，但模块里从未用到 {name!r}",
                )
            )
        return issues

    @staticmethod
    def _pain_point(
        relative: str,
        lineno: int,
        column: int,
        kind: str,
        severity: str,
        message: str,
    ) -> PainPoint:
        return PainPoint(
            id=f"{relative}:{lineno}:{kind}",
            type=kind,
            severity=severity,
            message=f"{message}（{relative}:{lineno}）",
            location=relative,
            context={"line": lineno, "column": column, "file": relative, "check": kind},
        )
