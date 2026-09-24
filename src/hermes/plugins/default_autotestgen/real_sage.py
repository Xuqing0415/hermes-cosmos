"""真实 Sage：只在能证明「import 什么」的时候才给补丁，否则明确拒绝。

写一行 ``from X import Y`` 是 trivial 的；难的是决定 X 和 Y。一个未定义的名字
``helper`` 可能来自项目内部、第三方依赖、标准库，也可能是 ``globals()`` 动态注入的
——猜错的结果是 patch 应用成功、测试却失败，甚至掩盖真正的错误。

所以这里是 oracle 而不是生成器：逐层收集**可证明**的候选，只有唯一候选时才落笔。

四层，从最有把握到最没把握：

===========  ==========================================  ==============================  ==========
层           判据                                         产出                            confidence
===========  ==========================================  ==============================  ==========
1 标准库模块  ``name in sys.stdlib_module_names``          ``import os``                   proven
2 标准库符号  在符号密集的 stdlib 模块里 ``hasattr`` 唯一    ``from collections import C``   unproven
3 项目内符号  扫描项目 ``.py`` 顶层 def/class/赋值，唯一      ``from pkg.utils import h``     proven
4 已声明依赖  读 pyproject/requirements + ``find_spec``     ``import click``                unproven
===========  ==========================================  ==============================  ==========

两条硬约束：

* **唯一匹配**——能证明的候选多于一个就拒绝，不猜、不投票；
* **宁缺毋滥**——没有候选就返回空 operations 的 PatchPlan，把理由写在 description 里
  （接口要求返回 PatchPlan，所以「返回 None」由空 operations 承载，下游 Knight 会拒绝执行）。

第 2、4 层产出的是「语法上正确、可导入」的绑定，但**意图**没有被证明（例如
``os.connect`` 确实存在，可项目想用的多半不是它），因此标为 unproven 并写明依据。
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

from hermes.core.interfaces import DomainContext, PainPoint, PatchPlan, Sage

from .real_perceiver import SKIP_DIRECTORIES


def _public_names(module_name: str, module: types.ModuleType) -> List[str]:
    """模块对外公开的名字。

    有 ``__all__`` 就照它来（标准库的权威声明）；没有时退化为「非下划线开头、且不是
    子模块」——这样既保住 ``math.pi`` / ``uuid.uuid4`` 这类真导出，又排除掉内部 import。
    """

    declared = getattr(module, "__all__", None)
    if declared is not None:
        return [str(item) for item in declared]

    found: List[str] = []
    for attribute in dir(module):
        if attribute.startswith("_"):
            continue
        try:
            value = getattr(module, attribute)
        except Exception:
            continue
        if isinstance(value, types.ModuleType):
            continue
        found.append(attribute)
    return found


CONFIDENCE_PROVEN = "proven"
CONFIDENCE_UNPROVEN = "unproven"

BASIS_STDLIB_MODULE = "stdlib_module"
BASIS_STDLIB_SYMBOL = "stdlib_symbol"
BASIS_PROJECT_SYMBOL = "project_symbol"
BASIS_DECLARED_DEPENDENCY = "declared_dependency"

REFUSAL_PREFIX = "[拒绝]"
PATCH_PREFIX = "[补丁]"

#: 第 2 层只扫「以符号为主」的标准库模块：`os` / `sys` / `logging` 这类模块通常通过
#: 属性访问使用（`os.path.join`），那种情况下未定义的是模块名，第 1 层就解决了；
#: 把它们放进来只会放大误报。宁可漏报。
STDLIB_SYMBOL_MODULES = (
    "abc",
    "base64",
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "io",
    "itertools",
    "json",
    "math",
    "operator",
    "pathlib",
    "re",
    "statistics",
    "string",
    "textwrap",
    "types",
    "typing",
    "unittest",
    "uuid",
    "weakref",
)

DEPENDENCY_FILES = ("pyproject.toml", "requirements.txt")
REQUIREMENT_SPLIT = re.compile(r"[\[\s<>=!~;]")
NAME_IN_MESSAGE = re.compile(r"'([^']+)'")


@dataclass(frozen=True)
class Resolution:
    """一条可证明的候选：语句 + 依据 + 可信程度。"""

    statement: str
    basis: str
    confidence: str


def _is_package_init(name: str) -> bool:
    return name == "__init__.py"


def _module_path(root: str, path: str) -> Optional[str]:
    """把 ``path`` 变成可导入的点分模块名；不可导入时返回 None。

    ``src/click/core.py`` -> ``click.core``；根目录下的 ``calc.py`` -> ``calc``；
    位于非包目录里的 ``tests/helpers.py`` -> None（宁可拒绝，也不给一个导不进来的名字）。
    """

    directory = os.path.dirname(os.path.abspath(path))
    root = os.path.abspath(root)
    collected: List[str] = []
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        collected.insert(0, os.path.basename(directory))
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent

    basename = os.path.basename(path)
    if _is_package_init(basename):
        return ".".join(collected) or None
    stem = basename[:-3]
    if collected:
        return ".".join(collected + [stem])
    if directory == root:
        return stem
    return None


class RealSage(Sage):
    """默认域的真实补丁生成：唯一匹配才出补丁，其余情况如实拒绝。"""

    def __init__(self) -> None:
        self._project_index: Dict[str, Dict[str, List[str]]] = {}
        self._stdlib_symbols: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------ 入口
    def generate_patch(self, context: DomainContext, pain_point: PainPoint) -> PatchPlan:
        if pain_point is None:
            return self._refuse("没有 pain point", "")
        if pain_point.type != "undefined_name":
            return self._refuse(f"没有针对 {pain_point.type} 的模板（Sage 只处理未定义名）", pain_point.id)

        name = self._name_of(pain_point)
        if not name:
            return self._refuse("无法确定要导入的名字", pain_point.id)
        if not name.isidentifier():
            return self._refuse(f"{name!r} 不是合法标识符", pain_point.id)

        candidates = self._candidates(context, pain_point, name)
        statements = sorted({item.statement for item in candidates})

        if not statements:
            return self._refuse(
                f"无法证明 {name!r} 的来源：项目内没有唯一导出它的模块，"
                "标准库与已声明的依赖里也没有可导入的同名目标",
                pain_point.id,
            )
        if len(statements) > 1:
            return self._refuse(
                f"{name!r} 有多个可证明的来源，拒绝猜测：{'、'.join(statements)}",
                pain_point.id,
            )

        winner = next(item for item in candidates if item.statement == statements[0])
        location = pain_point.location
        if not location:
            return self._refuse(f"pain point 没有文件位置，无法插入 {statements[0]}", pain_point.id)

        after_line = self._insert_after(context, location)
        operation = {
            "type": "insert_lines",
            "file": location,
            "after_line": after_line,
            "lines": [winner.statement],
            "statement": winner.statement,
            "basis": winner.basis,
            "confidence": winner.confidence,
        }
        return PatchPlan(
            id=f"{pain_point.id}:import",
            pain_point_id=pain_point.id,
            description=(
                f"{PATCH_PREFIX} 在 {location} 第 {after_line} 行后插入 {winner.statement}"
                f"（依据：{winner.basis}，可信度：{winner.confidence}）"
            ),
            operations=[operation],
            estimated_effort=0.1,
        )

    # ------------------------------------------------------------------ 四层
    def _candidates(self, context: DomainContext, pain_point: PainPoint, name: str) -> List[Resolution]:
        found: List[Resolution] = []

        # 第 1 层：标准库顶层模块
        if name in getattr(sys, "stdlib_module_names", frozenset()):
            found.append(Resolution(f"import {name}", BASIS_STDLIB_MODULE, CONFIDENCE_PROVEN))

        # 第 2 层：标准库导出符号（有界扫描 + hasattr 命中）
        for module in self._stdlib_symbols_for(name):
            found.append(Resolution(f"from {module} import {name}", BASIS_STDLIB_SYMBOL, CONFIDENCE_UNPROVEN))

        # 第 3 层：项目内唯一导出符号
        for module in self._project_modules_for(context.path, name, pain_point.location):
            found.append(Resolution(f"from {module} import {name}", BASIS_PROJECT_SYMBOL, CONFIDENCE_PROVEN))

        # 第 4 层：已声明的依赖里可导入的顶层模块
        if self._is_importable_dependency(context.path, name):
            found.append(Resolution(f"import {name}", BASIS_DECLARED_DEPENDENCY, CONFIDENCE_UNPROVEN))

        return found

    def _stdlib_symbols_for(self, name: str) -> List[str]:
        matches = self._stdlib_symbol_index().get(name, [])
        if len(matches) <= 1:
            return matches
        # 同一个对象常被多个模块再导出（`Counter` 同时在 collections 与 typing 的公开名里）。
        # 这不是「多个来源」，而是「一个定义 + 若干再导出」：定义它的那个模块才是权威答案。
        definers = [module for module in matches if self._defines(module, name)]
        return definers or matches

    @staticmethod
    def _defines(module_name: str, name: str) -> bool:
        module = sys.modules.get(module_name)
        if module is None:
            return False
        value = getattr(module, name, None)
        # typing 里的 Counter/Dict/List 是泛型别名：真身在 __origin__，不是这个别名本身。
        # 取真身再问「它是谁定义的」，只有真正的定义者能匹配自己的名字。
        origin = getattr(value, "__origin__", None)
        subject = origin if origin is not None else value
        return getattr(subject, "__module__", None) == module_name

    def _stdlib_symbol_index(self) -> Dict[str, List[str]]:
        """``符号 -> 导出它的标准库模块``，只用各模块**对外公开**的名字。

        不能直接用 ``hasattr``：``contextlib`` / ``pathlib`` / ``uuid`` 内部都 ``import os``，
        于是 ``hasattr(contextlib, "os")`` 为真——那是实现细节，不是导出。把它们当候选会
        让明明唯一正确的 ``import os`` 变成「多个来源」，白白拒绝一个能证明的补丁。
        """

        if self._stdlib_symbols is not None:
            return self._stdlib_symbols

        index: Dict[str, List[str]] = {}
        for module_name in STDLIB_SYMBOL_MODULES:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for public in _public_names(module_name, module):
                index.setdefault(public, []).append(module_name)
        self._stdlib_symbols = index
        return index

    def _project_modules_for(self, root: str, name: str, location: Optional[str]) -> List[str]:
        index = self._project_index.get(root)
        if index is None:
            index = self._build_project_index(root)
            self._project_index[root] = index

        modules = index.get(name, [])
        own_module = self._own_module(root, location)
        return sorted(module for module in modules if module != own_module)

    @staticmethod
    def _own_module(root: str, location: Optional[str]) -> Optional[str]:
        if not location:
            return None
        path = os.path.join(root, location.replace("/", os.sep))
        return _module_path(root, path)

    @staticmethod
    def _build_project_index(root: str) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for base, directories, files in os.walk(root):
            directories[:] = sorted(name for name in directories if name not in SKIP_DIRECTORIES)
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(base, name)
                module = _module_path(root, path)
                if not module:
                    continue
                for symbol in _top_level_symbols(path):
                    index.setdefault(symbol, []).append(module)
        return index

    @staticmethod
    def _is_importable_dependency(root: str, name: str) -> bool:
        declared = _declared_dependencies(root)
        if name not in declared and name.replace("_", "-") not in declared:
            return False
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, AttributeError):
            return False

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _name_of(pain_point: PainPoint) -> Optional[str]:
        metadata = pain_point.context or {}
        name = metadata.get("name")
        if isinstance(name, str) and name:
            return name
        match = NAME_IN_MESSAGE.search(pain_point.message or "")
        return match.group(1) if match else None

    @staticmethod
    def _insert_after(context: DomainContext, location: str) -> int:
        """插入点：最后一条顶层 import 之后；没有 import 就放在模块 docstring 之后。"""

        path = os.path.join(context.path, location.replace("/", os.sep))
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
        except (OSError, SyntaxError, ValueError):
            return 0

        after = 0
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                after = max(after, statement.end_lineno or statement.lineno)
        if after:
            return after
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            return tree.body[0].end_lineno or tree.body[0].lineno
        return 0

    @staticmethod
    def _refuse(reason: str, pain_point_id: str) -> PatchPlan:
        return PatchPlan(
            id=f"{pain_point_id}:refused" if pain_point_id else "refused",
            pain_point_id=pain_point_id,
            description=f"{REFUSAL_PREFIX} {reason}",
            operations=[],
            estimated_effort=0.0,
        )


def _top_level_symbols(path: str) -> List[str]:
    """模块顶层的 def / class / 赋值目标名（用于第 3 层的符号索引）。"""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, ValueError):
        return []

    symbols: List[str] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    symbols.append(target.id)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            symbols.append(statement.target.id)
    return symbols


def _declared_dependencies(root: str) -> List[str]:
    """从 pyproject.toml / requirements*.txt 提取声明过的依赖名（去版本、extras、marker）。"""

    names: List[str] = []
    names.extend(_dependencies_from_pyproject(os.path.join(root, "pyproject.toml")))
    for filename in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if filename.startswith("requirements") and filename.endswith(".txt"):
            names.extend(_dependencies_from_requirements(os.path.join(root, filename)))
    return names


def _dependencies_from_requirements(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    names: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-", "git+", "http")):
            continue
        names.append(REQUIREMENT_SPLIT.split(stripped)[0].strip())
    return [name for name in names if name]


def _dependencies_from_pyproject(path: str) -> List[str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        return []
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return []

    names: List[str] = []
    project = data.get("project") or {}
    for entry in project.get("dependencies") or []:
        if isinstance(entry, str):
            names.append(REQUIREMENT_SPLIT.split(entry)[0].strip())

    poetry = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
    for key, value in poetry.items():
        if key.lower() != "python" and not isinstance(value, dict):
            names.append(key.replace("-", "_"))
        elif key.lower() != "python" and isinstance(value, dict) and "version" in value:
            names.append(key.replace("-", "_"))
    return [name for name in names if name]
