"""默认（Python）域插件：把真实实现与演示假桩分开摆放。

``mode="real"``（默认）使用真实实现：:class:`RealPerceiver` 基于 AST 做静态检查并报出
真实文件位置，:class:`RealKnight` 真的运行项目测试、失败就返回失败。

``mode="stub"`` 保留旧的演示行为——固定报 ``api/handler.py``、无条件 ``success=True``。
它只用于对比与回归测试，**不参与任何结论**：真实缺陷基准测到的 ``false_claim_rate``
就是这类假桩造成的。

真实 Sage 只在能证明「import 什么」时才出补丁（见 ``real_sage.py`` 的四层 oracle），
其余情况返回空 operations 并写明理由，由 Knight 如实拒绝执行。
"""

import hashlib
import time
from typing import List

from hermes.core.interfaces import (
    DomainContext,
    DomainPlugin,
    ExecutionResult,
    Knight,
    PainPoint,
    PatchPlan,
    Perceiver,
    Sage,
)

from .real_knight import RealKnight
from .real_perceiver import RealPerceiver
from .real_sage import RealSage

MODE_REAL = "real"
MODE_STUB = "stub"
KNOWN_MODES = (MODE_REAL, MODE_STUB)


class StubPerceiver(Perceiver):
    """演示假桩：报出固定位置，与实际代码无关。仅用于对比与回归测试。"""

    def detect(self, context: DomainContext) -> List[PainPoint]:
        pain_points = []

        sample_issues = [
            {
                "id": "1",
                "type": "null_pointer",
                "severity": "high",
                "message": "Potential NullPointerException in API handler",
                "location": "api/handler.py",
                "context": {"function": "handle_request", "line": 45},
            },
            {
                "id": "2",
                "type": "missing_validation",
                "severity": "medium",
                "message": "Missing input validation for request body",
                "location": "api/handler.py",
                "context": {"function": "handle_request", "line": 50},
            },
            {
                "id": "3",
                "type": "performance",
                "severity": "low",
                "message": "Potential performance bottleneck in data processing",
                "location": "utils/processor.py",
                "context": {"function": "process_batch", "line": 120},
            },
        ]

        for issue in sample_issues:
            pain_points.append(
                PainPoint(
                    id=issue["id"],
                    type=issue["type"],
                    severity=issue["severity"],
                    message=issue["message"],
                    location=issue["location"],
                    context=issue["context"],
                )
            )

        return pain_points


class StubSage(Sage):
    """模板桩：只有演示用的三种 pain point 有模板，其余类型生成 0 个操作。"""

    def generate_patch(self, context: DomainContext, pain_point: PainPoint) -> PatchPlan:
        operations = []

        if pain_point.type == "null_pointer":
            operations.append(
                {
                    "type": "add_null_check",
                    "file": pain_point.location or "api/handler.py",
                    "line": pain_point.context.get("line", 45),
                    "code": 'if request.body is None:\n    return error_response("Null body not allowed")',
                }
            )
        elif pain_point.type == "missing_validation":
            operations.append(
                {
                    "type": "add_validation",
                    "file": pain_point.location or "api/handler.py",
                    "line": pain_point.context.get("line", 50),
                    "code": "validate_input(request.body)",
                }
            )
        elif pain_point.type == "performance":
            operations.append(
                {
                    "type": "optimize_loop",
                    "file": pain_point.location or "utils/processor.py",
                    "line": pain_point.context.get("line", 120),
                    "code": "result = parallel_process(items)",
                }
            )

        patch_id = hashlib.md5(f"{pain_point.id}_{int(time.time())}".encode()).hexdigest()

        return PatchPlan(
            id=patch_id,
            pain_point_id=pain_point.id,
            description=f"Fix for {pain_point.type}: {pain_point.message}",
            operations=operations,
            estimated_effort=0.5,
        )


class StubKnight(Knight):
    """演示假桩：无条件返回成功，测试是否真的跑过它并不关心。仅用于对比与回归测试。"""

    def execute(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        verification_results = []

        for op in patch_plan.operations:
            verification_results.append(
                {
                    "operation": op["type"],
                    "file": op["file"],
                    "status": "applied",
                    "message": f"Patch applied to {op['file']}",
                }
            )

        return ExecutionResult(
            success=True,
            patch_plan_id=patch_plan.id,
            message="All operations executed successfully",
            verification_results=verification_results,
            artifacts={"patch_file": f"{patch_plan.id}.patch"},
        )

    def verify(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        verification_results = [
            {"test": "unit_test", "status": "passed", "duration_ms": 120},
            {"test": "integration_test", "status": "passed", "duration_ms": 450},
            {"test": "regression_test", "status": "passed", "duration_ms": 280},
        ]

        return ExecutionResult(
            success=True,
            patch_plan_id=patch_plan.id,
            message="All verification tests passed",
            verification_results=verification_results,
        )


class DefaultAutotestGenPlugin(DomainPlugin):
    def __init__(self, mode: str = MODE_REAL):
        if mode not in KNOWN_MODES:
            raise ValueError(f"未知 mode：{mode!r}，可选 {KNOWN_MODES}")
        self.mode = mode

    @property
    def domain_name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Default AutoTestGen plugin for Python project testing and verification"

    def create_context(self, path: str, **kwargs) -> DomainContext:
        return DomainContext(domain="default", path=path, metadata=kwargs, issues=[], artifacts={})

    def get_perceiver(self) -> Perceiver:
        return RealPerceiver() if self.mode == MODE_REAL else StubPerceiver()

    def get_sage(self) -> Sage:
        return RealSage() if self.mode == MODE_REAL else StubSage()

    def get_knight(self) -> Knight:
        return RealKnight() if self.mode == MODE_REAL else StubKnight()
