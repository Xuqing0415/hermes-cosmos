"""默认（Python）域的真实执行者：真的写文件、真的跑测试，失败就返回失败。

旧实现（``StubKnight``）无条件返回 ``success=True`` 并打印固定的
``unit_test/integration_test/regression_test = passed``——那是整个系统里最危险的一处：
它让「系统自称成功」与「缺陷真的被修好」彻底脱钩。本模块把这两件事重新绑在一起：

* :meth:`RealKnight.verify` 真的在目标目录里跑 ``pytest``，返回真实退出码、真实状态与
  原始输出；超时、跑不起来都如实标记为未通过；
* :meth:`RealKnight.execute` 只落地它认得且能验证的操作（目前只有 Sage 产出的
  ``insert_lines``），落地后**重新解析文件**确认没写坏，再跑测试；任何一步失败都返回
  失败。不认识的操作一律拒绝，绝不假装写入。

关于「改哪里」——默认**绝不原地改**：

* ``execute(context, plan)`` 会把 ``context.path`` 复制到隔离目录，在副本里应用补丁并跑
  测试，然后删掉副本（``artifacts["isolated"] is True``）；隔离目录建不出来时直接返回
  失败，而不是退化成原地修改；
* ``execute(context, plan, apply_to=...)`` 由调用方明确交出某个目录的写权限（基准测试
  就是这么做的：它把补丁落到自己刚创建的一次性 git worktree 里，然后**独立地**重跑测试
  来判定系统到底有没有修好——判据不经过系统自己）。
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hermes.core.interfaces import DomainContext, ExecutionResult, Knight, PatchPlan

DEFAULT_TIMEOUT = 120
EVIDENCE_LIMIT = 4000
SUPPORTED_OPERATION = "insert_lines"

#: 复制隔离副本时跳过的目录：这些不含被测源码，复制它们只会慢
COPY_SKIP = (
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".eggs",
)


@dataclass
class _Target:
    root: str
    isolated: bool
    cleanup: Optional[str] = None


class RealKnight(Knight):
    """真实执行 + 真实验证。没有任何一条路径会无条件返回成功。"""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, interpreter: Optional[str] = None):
        self.timeout = max(1, timeout)
        self.interpreter = interpreter or sys.executable

    # ------------------------------------------------------------------ 执行
    def execute(
        self,
        context: DomainContext,
        patch_plan: PatchPlan,
        apply_to: Optional[str] = None,
    ) -> ExecutionResult:
        patch_plan_id = patch_plan.id if patch_plan else ""
        operations = list(patch_plan.operations or []) if patch_plan else []
        if not operations:
            return ExecutionResult(
                success=False,
                patch_plan_id=patch_plan_id,
                message="没有可执行的操作：Sage 没有生成补丁（它要么拒绝了，要么没有模板），Knight 不假装执行",
                verification_results=[],
                artifacts={"applied_files": [], "isolated": None},
            )

        target = self._target_root(context, apply_to)
        if target is None:
            return ExecutionResult(
                success=False,
                patch_plan_id=patch_plan_id,
                message="无法创建隔离工作目录：拒绝在原地修改目标项目",
                verification_results=[],
                artifacts={"applied_files": [], "isolated": True},
            )

        try:
            entries, applied_files = self._apply_operations(target.root, operations)
            tests = self._run_pytest(target.root, patch_plan_id, self._tmp_dir(context))
            passed = tests.success
            failure = any(entry["status"] != "applied" for entry in entries)
            success = bool(applied_files) and not failure and passed
            return ExecutionResult(
                success=success,
                patch_plan_id=patch_plan_id,
                message=self._message(entries, applied_files, passed),
                verification_results=entries + tests.verification_results,
                artifacts={
                    "root": target.root,
                    "isolated": target.isolated,
                    "applied_files": applied_files,
                    "confidence": self._confidences(operations),
                    **{key: value for key, value in tests.artifacts.items() if key != "applied_files"},
                },
            )
        finally:
            if target.cleanup:
                shutil.rmtree(target.cleanup, ignore_errors=True)

    @staticmethod
    def _confidences(operations: List[Dict[str, Any]]) -> List[str]:
        return [str(operation.get("confidence", "unknown")) for operation in operations]

    @staticmethod
    def _message(entries: List[Dict[str, Any]], applied_files: List[str], passed: bool) -> str:
        if not applied_files:
            reasons = "；".join(str(entry.get("message", "")) for entry in entries)
            return f"没有任何操作被落地：{reasons}"
        verdict = "测试通过" if passed else "测试未通过"
        return f"已落地 {len(applied_files)} 个文件（{', '.join(applied_files)}），{verdict}"

    # ------------------------------------------------------------------ 落地
    def _apply_operations(self, root: str, operations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        entries: List[Dict[str, Any]] = []
        applied_files: List[str] = []
        for operation in operations:
            entry, applied = self._apply_one(root, operation)
            entries.append(entry)
            if applied and entry["file"] not in applied_files:
                applied_files.append(entry["file"])
        return entries, applied_files

    def _apply_one(self, root: str, operation: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        operation_type = str(operation.get("type", "unknown"))
        relative = str(operation.get("file") or operation.get("path") or "")
        entry: Dict[str, Any] = {"operation": operation_type, "file": relative}

        if not relative:
            entry.update(status="skipped", message="操作没有指定目标文件")
            return entry, False
        path = os.path.join(root, relative.replace("/", os.sep))
        if not os.path.isfile(path):
            entry.update(status="failed", message=f"目标文件不存在：{relative}（Knight 不会凭空创建文件）")
            return entry, False
        if operation_type != SUPPORTED_OPERATION:
            entry.update(
                status="refused",
                message=f"不认识的操作类型 {operation_type}（只支持 {SUPPORTED_OPERATION}），拒绝猜测落地方式",
            )
            return entry, False

        reason = self._insert_lines(path, operation)
        if reason:
            entry.update(status="failed", message=reason)
            return entry, False
        entry.update(status="applied", message=f"已在 {relative} 插入 {len(operation.get('lines') or [])} 行")
        return entry, True

    @staticmethod
    def _insert_lines(path: str, operation: Dict[str, Any]) -> Optional[str]:
        """把若干行插入文件的指定位置；写坏了就还原并返回原因。"""

        lines = [str(item) for item in (operation.get("lines") or [])]
        if not lines:
            return "操作为空：没有要插入的行"
        after_line = int(operation.get("after_line") or 0)

        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                original = handle.read()
        except (OSError, UnicodeDecodeError) as error:
            return f"无法读取 {path}：{error}"

        updated = _splice(original, after_line, lines)
        try:
            ast.parse(updated, filename=path)
        except SyntaxError as error:
            # 插入把文件写坏了：这是应用逻辑的错，不是缺陷的错，必须还原并如实报告
            return f"插入后文件无法解析（{error.msg}，第 {error.lineno} 行），已放弃修改"

        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
        except OSError as error:
            return f"无法写入 {path}：{error}"
        return None

    # ------------------------------------------------------------------ 隔离
    def _target_root(self, context: DomainContext, apply_to: Optional[str]) -> Optional[_Target]:
        if apply_to:
            return _Target(root=os.path.abspath(apply_to), isolated=False)

        source = os.path.abspath(context.path or ".")
        metadata = context.metadata or {}
        base = metadata.get("isolation_root") or os.environ.get("HERMES_ISOLATION_ROOT") or tempfile.gettempdir()
        created = os.path.join(base, f"hermes_knight_{uuid.uuid4().hex[:12]}")
        try:
            # 自己建目录而不是用 tempfile.mkdtemp：受限环境（本项目的 CI 沙箱就是）里
            # mkdtemp 建出来的目录可能拒绝写入，那样隔离就等于没做
            os.makedirs(created)
            target = os.path.join(created, os.path.basename(source) or "target")
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(*COPY_SKIP))
        except OSError:
            shutil.rmtree(created, ignore_errors=True)
            return None
        return _Target(root=target, isolated=True, cleanup=created)

    # ------------------------------------------------------------------ 验证
    def verify(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        return self._run_pytest(context.path or ".", patch_plan.id if patch_plan else "", self._tmp_dir(context))

    @staticmethod
    def _tmp_dir(context: DomainContext) -> Optional[str]:
        value = (context.metadata or {}).get("pytest_tmpdir")
        return str(value) if value else None

    def _run_pytest(self, root: str, patch_plan_id: str, tmp_dir: Optional[str] = None) -> ExecutionResult:
        collected = self._pytest_targets(root)
        command = [self.interpreter, "-m", "pytest", *collected, "-q", "--tb=no", "-p", "no:cacheprovider"]
        started = time.time()

        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._pytest_env(root, tmp_dir),
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._failure(
                patch_plan_id,
                command,
                root,
                started,
                status="timeout",
                message=f"测试超时（>{self.timeout}s），无法判定为通过",
            )
        except OSError as error:
            return self._failure(
                patch_plan_id,
                command,
                root,
                started,
                status="error",
                message=f"无法运行测试：{error}",
            )

        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        passed = completed.returncode == 0
        return ExecutionResult(
            success=passed,
            patch_plan_id=patch_plan_id,
            message=f"pytest 退出码 {completed.returncode}：{'测试全部通过' if passed else '存在失败，未通过验证'}",
            verification_results=[
                {
                    "test": "pytest",
                    "status": "passed" if passed else "failed",
                    "returncode": completed.returncode,
                    "duration_ms": int((time.time() - started) * 1000),
                }
            ],
            artifacts={
                "command": " ".join(command),
                "cwd": root,
                "collected": collected,
                "exit_code": completed.returncode,
                "output_tail": output[-EVIDENCE_LIMIT:],
            },
        )

    @staticmethod
    def _pytest_targets(root: str) -> List[str]:
        """明确指定收集目录，只跑目标项目自己的测试。

        不指定时，若目标目录自己没有 pytest 配置，pytest 会一路向上找到**别的**项目的
        ``pyproject.toml``，把那个项目的 ``testpaths`` 当成本项目的测试范围——那就测错对象了。
        """

        for candidate in ("tests", "test"):
            if os.path.isdir(os.path.join(root, candidate)):
                return [candidate]
        return ["."]

    @staticmethod
    def _pytest_env(root: str, tmp_dir: Optional[str]) -> Dict[str, str]:
        """让源码布局的项目能被导入，并把临时目录指向可写位置。

        ``PYTHONPATH`` 带上仓库根与 ``src/``：从源码检出直接跑测试的标准做法，与基准自身的
        TestRunner 保持一致。``TEMP``/``TMP`` 在受限环境里可能不可写，一碰 ``tempfile`` 就会
        PermissionError——那是环境问题，不该算成缺陷没修好。
        """

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(root, "src"), root, env.get("PYTHONPATH", "")]).strip(
            os.pathsep
        )
        env["PYTHONIOENCODING"] = "utf-8"
        if tmp_dir:
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except OSError:
                return env
            for key in ("TEMP", "TMP", "TMPDIR"):
                env[key] = tmp_dir
        return env

    @staticmethod
    def _failure(
        patch_plan_id: str,
        command: List[str],
        root: str,
        started: float,
        status: str,
        message: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            patch_plan_id=patch_plan_id,
            message=message,
            verification_results=[
                {
                    "test": "pytest",
                    "status": status,
                    "duration_ms": int((time.time() - started) * 1000),
                }
            ],
            artifacts={"command": " ".join(command), "cwd": root, "exit_code": None, "output_tail": ""},
        )


def _splice(original: str, after_line: int, lines: List[str]) -> str:
    """在 ``after_line`` 之后插入 ``lines``，保留原文件的换行风格。"""

    parts = original.splitlines(keepends=True)
    ending = "\r\n" if any(part.endswith("\r\n") for part in parts) else "\n"
    if after_line > 0:
        index = min(after_line, len(parts))
    else:
        index = _first_code_index(parts)
    block = [f"{line}{ending}" for line in lines]
    return "".join(parts[:index]) + "".join(block) + "".join(parts[index:])


def _first_code_index(parts: List[str]) -> int:
    """跳过开头的空行与注释行（shebang、编码声明也在这里），从第一行真正代码前插入。"""

    index = 0
    while index < len(parts):
        stripped = parts[index].strip()
        if stripped and not stripped.startswith("#"):
            break
        index += 1
    return index
