"""默认（Python）域的真实执行者：真的跑测试，失败就返回失败。

旧实现（``StubKnight``）无条件返回 ``success=True`` 并打印固定的
``unit_test/integration_test/regression_test = passed``——那是整个系统里最危险的一处：
它让「系统自称成功」与「缺陷真的被修好」彻底脱钩。本模块把这两件事重新绑在一起：

* :meth:`RealKnight.verify` 真的在目标目录里跑 ``pytest``，返回真实退出码、真实状态与
  原始输出；超时、跑不起来都如实标记为未通过；
* :meth:`RealKnight.execute` 只落地能被证明安全的操作。当 Sage 给不出可应用的补丁内容
  （当前的模板 Sage 对真实 pain point 就是这样）时，它**拒绝执行并返回失败**，
  而不是假装写入了文件。它不写任何文件，因此也不产生副作用。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from hermes.core.interfaces import DomainContext, ExecutionResult, Knight, PatchPlan

DEFAULT_TIMEOUT = 120
EVIDENCE_LIMIT = 4000


class RealKnight(Knight):
    """真实执行 + 真实验证。没有任何一条路径会无条件返回成功。"""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, interpreter: Optional[str] = None):
        self.timeout = max(1, timeout)
        self.interpreter = interpreter or sys.executable

    # ------------------------------------------------------------------ 执行
    def execute(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        patch_plan_id = patch_plan.id if patch_plan else ""
        operations = list(patch_plan.operations or []) if patch_plan else []
        if not operations:
            return ExecutionResult(
                success=False,
                patch_plan_id=patch_plan_id,
                message="没有可执行的操作：Sage 没有为这个 pain point 生成任何操作，Knight 不假装执行",
                verification_results=[],
                artifacts={"applied_files": []},
            )

        results: List[Dict[str, Any]] = []
        for operation in operations:
            results.append(self._refuse(operation, context))

        return ExecutionResult(
            success=False,
            patch_plan_id=patch_plan_id,
            message=f"{len(results)} 个操作都没有被落地：没有可应用的补丁内容，Knight 拒绝猜测如何写入文件",
            verification_results=results,
            artifacts={"applied_files": []},
        )

    @staticmethod
    def _refuse(operation: Dict[str, Any], context: DomainContext) -> Dict[str, Any]:
        operation_type = str(operation.get("type", "unknown"))
        target = str(operation.get("file") or operation.get("path") or "")
        entry: Dict[str, Any] = {"operation": operation_type, "file": target}
        if not target:
            entry.update(status="skipped", message="操作没有指定目标文件，无法判断该改哪里")
            return entry
        if not os.path.isfile(os.path.join(context.path, target)):
            entry.update(status="failed", message=f"目标文件不存在：{target}（Knight 不会凭空创建文件）")
            return entry
        entry.update(
            status="refused",
            message="操作没有携带可应用的补丁内容（既不是统一 diff，也不是完整文件内容），拒绝猜测落地方式",
        )
        return entry

    # ------------------------------------------------------------------ 验证
    def verify(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        patch_plan_id = patch_plan.id if patch_plan else ""
        command = [self.interpreter, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider"]
        cwd = context.path or "."
        started = time.time()

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._failure(
                patch_plan_id,
                command,
                started,
                status="timeout",
                message=f"测试超时（>{self.timeout}s），无法判定为通过",
            )
        except OSError as error:
            return self._failure(
                patch_plan_id,
                command,
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
                "cwd": cwd,
                "exit_code": completed.returncode,
                "output_tail": output[-EVIDENCE_LIMIT:],
            },
        )

    @staticmethod
    def _failure(
        patch_plan_id: str,
        command: List[str],
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
            artifacts={"command": " ".join(command), "exit_code": None, "output_tail": ""},
        )
