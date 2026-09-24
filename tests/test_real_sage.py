"""真实 Sage 的 oracle 测试：只在能证明「导入什么」时才出补丁，否则拒绝。

每个用例都是一条完整的证据链：Perceiver 报出真实位置 → Sage 给出（或拒绝给）补丁 →
Knight 真的写进文件 → **项目自己的 pytest** 判定修好没有。

顺序很重要：先用合成用例证明它不撒谎，再拿真实历史检验它能不能修。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import uuid

import pytest

from hermes.core.interfaces import PainPoint, PatchPlan
from hermes.core.plugin_manager import PluginManager
from hermes.plugins.default_autotestgen import RealKnight, RealSage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def work_dir():
    base = os.path.join(REPO_ROOT, ".tmp_test", "real_sage")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _source(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _write(root: pathlib.Path, relative: str, text: str) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(work_dir: pathlib.Path, app_source: str, extra: dict | None = None) -> pathlib.Path:
    """一个最小的真实项目：一个坏掉的 app.py、一个始终通过的测试、一个项目内符号。"""

    root = work_dir / "project"
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/utils.py", _source("def helper(value):", "    return value * 2"))
    _write(root, "app.py", app_source)
    _write(root, "tests/test_other.py", _source("def test_other():", "    assert True"))
    for relative, text in (extra or {}).items():
        _write(root, relative, text)
    return root


def _pipeline(project: pathlib.Path, **metadata):
    plugin = PluginManager().get_plugin("default")
    context = plugin.create_context(str(project), **metadata)
    return plugin, context, plugin.get_perceiver().detect(context)


def _plan_of(plugin, context, pain_points):
    return plugin.get_sage().generate_patch(context, pain_points[0])


class TestOracleResolves:
    """四条合成用例：三条应当被解析出来，一条应当被拒绝，而且都要真的把测试跑绿。"""

    def test_stdlib_module_is_imported(self, work_dir):
        project = _project(
            work_dir,
            _source("def join_paths(first, second):", "    return os.path.join(first, second)"),
            {
                "tests/test_app.py": _source(
                    "from app import join_paths",
                    "",
                    "",
                    "def test_join():",
                    "    assert join_paths('a', 'b').replace('\\\\', '/') == 'a/b'",
                )
            },
        )
        plugin, context, pain_points = _pipeline(project)

        assert [point.context["name"] for point in pain_points] == ["os"]
        plan = _plan_of(plugin, context, pain_points)
        assert plan.operations[0]["statement"] == "import os"
        assert plan.operations[0]["basis"] == "stdlib_module"
        assert plan.operations[0]["confidence"] == "proven"

        assert _verify_fails(project)  # 补丁之前：测试确实红
        execution = RealKnight().execute(context, plan, apply_to=str(project))

        assert execution.success is True
        assert execution.artifacts["applied_files"] == ["app.py"]
        assert "import os" in (project / "app.py").read_text(encoding="utf-8")
        assert RealKnight().verify(context, plan).success is True  # 修好且没有引入新失败

    def test_stdlib_symbol_is_imported_from_its_module(self, work_dir):
        project = _project(
            work_dir,
            _source("def count_letters(text):", "    return Counter(text)"),
            {
                "tests/test_app.py": _source(
                    "from app import count_letters",
                    "",
                    "",
                    "def test_count():",
                    "    assert count_letters('aab')['a'] == 2",
                )
            },
        )
        plugin, context, pain_points = _pipeline(project)

        plan = _plan_of(plugin, context, pain_points)

        assert plan.operations[0]["statement"] == "from collections import Counter"
        assert plan.operations[0]["basis"] == "stdlib_symbol"
        # 语法上可导入，但「想用的是它」没被证明：如实标为 unproven
        assert plan.operations[0]["confidence"] == "unproven"
        assert RealKnight().execute(context, plan, apply_to=str(project)).success is True
        assert RealKnight().verify(context, plan).success is True

    def test_project_symbol_is_imported_from_the_module_that_defines_it(self, work_dir):
        project = _project(
            work_dir,
            _source("def doubled(values):", "    return [helper(value) for value in values]"),
            {
                "tests/test_app.py": _source(
                    "from app import doubled",
                    "",
                    "",
                    "def test_doubled():",
                    "    assert doubled([1, 2]) == [2, 4]",
                )
            },
        )
        plugin, context, pain_points = _pipeline(project)

        plan = _plan_of(plugin, context, pain_points)

        assert plan.operations[0]["statement"] == "from pkg.utils import helper"
        assert plan.operations[0]["basis"] == "project_symbol"
        assert plan.operations[0]["confidence"] == "proven"
        assert RealKnight().execute(context, plan, apply_to=str(project)).success is True
        assert RealKnight().verify(context, plan).success is True

    def test_unknown_name_is_refused_with_a_reason(self, work_dir):
        project = _project(work_dir, _source("def compute():", "    return foo_bar_baz(1)"))
        plugin, context, pain_points = _pipeline(project)

        plan = _plan_of(plugin, context, pain_points)

        assert plan.operations == []
        assert plan.description.startswith("[拒绝]")
        assert "无法证明" in plan.description
        execution = RealKnight().execute(context, plan, apply_to=str(project))
        assert execution.success is False
        assert execution.artifacts["applied_files"] == []

    def test_ambiguous_symbol_is_refused_not_guessed(self, work_dir):
        project = _project(
            work_dir,
            _source("def doubled(values):", "    return [helper(value) for value in values]"),
            {
                "tools/__init__.py": "",
                "tools/legacy.py": _source("def helper(value):", "    return value + 1"),
            },
        )
        plugin, context, pain_points = _pipeline(project)

        plan = _plan_of(plugin, context, pain_points)

        assert plan.operations == []
        assert "多个可证明的来源" in plan.description
        assert "pkg.utils" in plan.description and "tools.legacy" in plan.description

    def test_sage_refuses_pain_point_types_it_has_no_template_for(self, work_dir):
        plugin = PluginManager().get_plugin("default")
        context = plugin.create_context(str(work_dir))

        plan = RealSage().generate_patch(context, _unused_import_pain_point())

        assert plan.operations == []
        assert "没有针对" in plan.description


class TestOraclePlacement:
    def test_import_is_inserted_after_the_last_existing_import(self, work_dir):
        project = _project(
            work_dir,
            _source(
                '"""模块文档。"""',
                "import json",
                "",
                "",
                "def dump(value):",
                '    return json.dumps({"sep": os.sep, "value": value})',
            ),
            {
                "tests/test_app.py": _source(
                    "import json",
                    "",
                    "from app import dump",
                    "",
                    "",
                    "def test_dump():",
                    "    assert json.loads(dump(1))['value'] == 1",
                )
            },
        )
        plugin, context, pain_points = _pipeline(project)

        plan = _plan_of(plugin, context, pain_points)

        assert plan.operations[0]["after_line"] == 2
        assert RealKnight().execute(context, plan, apply_to=str(project)).success is True

        lines = (project / "app.py").read_text(encoding="utf-8").splitlines()
        assert lines[0] == '"""模块文档。"""'  # 文档字符串仍然是第一句
        assert lines.index("import json") == 1
        assert lines.index("import os") > lines.index("import json")
        assert RealKnight().verify(context, plan).success is True


class TestKnightSafety:
    def test_broken_insertion_is_undone_and_reported(self, work_dir):
        project = _project(work_dir, _source("VALUE = 1"))
        path = project / "app.py"
        before = path.read_text(encoding="utf-8")
        operation = {"type": "insert_lines", "file": "app.py", "after_line": 0, "lines": ["def ("]}

        entry, applied = RealKnight()._apply_one(str(project), operation)

        assert applied is False
        assert entry["status"] == "failed"
        assert "无法解析" in entry["message"]
        assert path.read_text(encoding="utf-8") == before  # 写坏了必须还原

    def test_unknown_operation_type_is_refused(self, work_dir):
        project = _project(work_dir, _source("VALUE = 1"))
        operation = {"type": "rewrite_everything", "file": "app.py"}

        entry, applied = RealKnight()._apply_one(str(project), operation)

        assert applied is False
        assert entry["status"] == "refused"
        assert (project / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_default_execution_never_touches_the_target_project(self, work_dir):
        project = _project(
            work_dir,
            _source("def doubled(values):", "    return [helper(value) for value in values]"),
            {
                "tests/test_app.py": _source(
                    "from app import doubled",
                    "",
                    "",
                    "def test_doubled():",
                    "    assert doubled([1, 2]) == [2, 4]",
                )
            },
        )
        isolation = work_dir / "isolation"
        isolation.mkdir()
        plugin, context, pain_points = _pipeline(project, isolation_root=str(isolation))
        plan = _plan_of(plugin, context, pain_points)
        before = (project / "app.py").read_text(encoding="utf-8")

        execution = RealKnight().execute(context, plan)  # 不给 apply_to：必须在副本里干活

        assert execution.artifacts["isolated"] is True
        assert execution.success is True  # 副本里测试是通过的
        assert (project / "app.py").read_text(encoding="utf-8") == before  # 原地一个字节都没动
        assert list(isolation.iterdir()) == []  # 副本用完即删

    def test_unwritable_isolation_root_fails_instead_of_writing_in_place(self, work_dir):
        project = _project(
            work_dir,
            _source("def doubled(values):", "    return [helper(value) for value in values]"),
        )
        # 把一个**文件**当作隔离根目录：建不出隔离副本
        blocked = _write(work_dir, "not_a_directory", "x\n")
        plugin, context, pain_points = _pipeline(project, isolation_root=str(blocked))
        plan = _plan_of(plugin, context, pain_points)
        before = (project / "app.py").read_text(encoding="utf-8")

        execution = RealKnight().execute(context, plan)

        assert execution.success is False
        assert "拒绝在原地修改" in execution.message
        assert (project / "app.py").read_text(encoding="utf-8") == before

    def test_no_plan_is_reported_as_nothing_to_do(self, work_dir):
        project = _project(work_dir, _source("VALUE = 1"))
        empty = PatchPlan(id="p", pain_point_id="pp", description="x", operations=[])

        execution = RealKnight().execute(_context(project), empty, apply_to=str(project))

        assert execution.success is False
        assert "没有可执行的操作" in execution.message


class TestAgainstRealHistory:
    """拿真实历史检验：系统补的 import 与人类当年补的，是不是同一个目标。"""

    SHA = "22c0503e"
    RELATIVE = "src/hermes/agent_civilization/civilization_migration.py"

    def test_oracle_agrees_with_the_import_a_human_committed(self, work_dir):
        parent = _git_show(f"{self.SHA}^:{self.RELATIVE}")
        fixed = _git_show(f"{self.SHA}:{self.RELATIVE}")
        if not parent or not fixed:
            pytest.skip("本仓库历史里没有这个提交（CI 检出的是浅克隆）")

        project = work_dir / "history"
        _write(project, self.RELATIVE, parent)
        plugin = PluginManager().get_plugin("default")
        context = plugin.create_context(str(project))

        pain_points = plugin.get_perceiver().detect(context)
        target = [point for point in pain_points if point.context.get("name") == "List"]
        assert target, "父提交状态里应当报出未定义名 List"

        plan = _plan_of(plugin, context, target)
        assert plan.operations[0]["statement"] == "from typing import List"

        human = [line for line in fixed.splitlines() if line.startswith("from typing import")]
        assert any("List" in line for line in human), "人类当年也是从 typing 导入 List"


def _context(project: pathlib.Path):
    return PluginManager().get_plugin("default").create_context(str(project))


def _verify_fails(project: pathlib.Path) -> bool:
    context = _context(project)
    plan = PatchPlan(id="probe", pain_point_id="probe", description="probe", operations=[])
    return RealKnight().verify(context, plan).success is False


def _unused_import_pain_point() -> PainPoint:
    return PainPoint(
        id="mod.py:1:unused_import",
        type="unused_import",
        severity="low",
        message="导入了 import json，但模块里从未用到 'json'（mod.py:1）",
        location="mod.py",
        context={"line": 1, "name": "json"},
    )


def _git_show(spec: str):
    completed = subprocess.run(
        ["git", "-C", REPO_ROOT, "show", spec],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout if completed.returncode == 0 else None
