"""默认（Python）域插件：真实实现必须报真位置，假桩只能留在 stub 模式里。

这里的每个断言都盯着一个具体问题：系统报出的东西有没有证据。
真实感知器的证据是 AST 节点，真实执行者的证据是 pytest 的退出码与输出。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import time
import uuid

import pytest

from hermes.core.interfaces import DomainContext, PatchPlan
from hermes.core.plugin_manager import PluginManager
from hermes.plugins.default_autotestgen import (
    DefaultAutotestGenPlugin,
    RealKnight,
    RealPerceiver,
    StubKnight,
    StubPerceiver,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def work_dir():
    base = os.path.join(REPO_ROOT, ".tmp_test", "default_autotestgen")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write(root: pathlib.Path, relative: str, text: str) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _context(root: pathlib.Path) -> DomainContext:
    return DomainContext(domain="default", path=str(root), metadata={}, issues=[], artifacts={})


class TestRealPerceiver:
    def test_reports_undefined_name_with_its_real_location(self, work_dir):
        _write(work_dir, "geometry.py", "def area(radius):\n    return pi * radius ** 2\n")

        issues = RealPerceiver().detect(_context(work_dir))

        assert len(issues) == 1
        issue = issues[0]
        assert issue.type == "undefined_name"
        assert issue.severity == "high"
        assert issue.location == "geometry.py"
        assert issue.context["line"] == 2
        assert "pi" in issue.message

    def test_reports_unused_import_at_its_real_line(self, work_dir):
        _write(work_dir, "mod.py", "import json\n\n\ndef f():\n    return 1\n")

        issues = RealPerceiver().detect(_context(work_dir))

        assert [issue.type for issue in issues] == ["unused_import"]
        assert issues[0].location == "mod.py"
        assert issues[0].context["line"] == 1
        assert issues[0].severity == "low"

    def test_clean_module_produces_no_claims_at_all(self, work_dir):
        _write(work_dir, "clean.py", "import json\n\n\ndef dump(x):\n    return json.dumps(x)\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_locals_arguments_and_builtins_are_not_reported(self, work_dir):
        _write(
            work_dir,
            "locals.py",
            "def total_of(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += len(item)\n"
            "    return total\n",
        )

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_future_imports_are_not_called_unused(self, work_dir):
        # 这是自查时抓到的一个误报：`annotations` 从来不会被当名字用
        _write(work_dir, "future.py", "from __future__ import annotations\n\n\ndef f(x: int) -> int:\n    return x\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_locations_are_root_relative_with_forward_slashes(self, work_dir):
        _write(work_dir, "pkg/sub/mod.py", "def f():\n    return missing_name\n")

        issues = RealPerceiver().detect(_context(work_dir))

        assert [issue.location for issue in issues] == ["pkg/sub/mod.py"]
        assert "\\" not in issues[0].location

    def test_virtualenv_and_cache_directories_are_skipped(self, work_dir):
        _write(work_dir, ".venv/lib/broken.py", "def f():\n    return nothing_here\n")
        _write(work_dir, "__pycache__/broken.py", "def f():\n    return nothing_here\n")
        _write(work_dir, "ok.py", "VALUE = 1\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_dunder_init_is_exempt_from_unused_import_reports(self, work_dir):
        _write(work_dir, "pkg/__init__.py", "from .core import Thing\n")
        _write(work_dir, "pkg/core.py", "class Thing:\n    pass\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_star_import_disables_undefined_name_claims(self, work_dir):
        # 导入内容不可知时，任何「未定义」判断都可能误报，因此整块放弃
        _write(work_dir, "star.py", "from os.path import *\n\n\ndef f():\n    return join('a', 'b')\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_unparseable_file_is_skipped_instead_of_guessed(self, work_dir):
        _write(work_dir, "broken.py", "def f(:\n    pass\n")

        assert RealPerceiver().detect(_context(work_dir)) == []

    def test_findings_are_deterministic(self, work_dir):
        _write(work_dir, "a.py", "def f():\n    return nope\n")
        _write(work_dir, "b.py", "import os\n")

        first = [issue.to_dict() for issue in RealPerceiver().detect(_context(work_dir))]
        second = [issue.to_dict() for issue in RealPerceiver().detect(_context(work_dir))]

        assert first == second
        assert [issue["id"] for issue in first] == ["a.py:2:undefined_name", "b.py:1:unused_import"]


class TestRealKnight:
    def test_verify_passes_only_when_the_tests_really_pass(self, work_dir):
        _write(work_dir, "tests/test_ok.py", "def test_ok():\n    assert True\n")

        result = RealKnight().verify(_context(work_dir), _plan())

        assert result.success is True
        assert result.verification_results[0]["status"] == "passed"
        assert result.artifacts["exit_code"] == 0

    def test_verify_reports_failure_with_the_real_evidence(self, work_dir):
        _write(work_dir, "tests/test_bad.py", "def test_bad():\n    assert 1 == 2\n")

        result = RealKnight().verify(_context(work_dir), _plan())

        assert result.success is False
        assert result.verification_results[0]["status"] == "failed"
        assert result.artifacts["exit_code"] != 0
        assert "test_bad" in result.artifacts["output_tail"]

    def test_verify_marks_a_timeout_as_not_passed(self, work_dir):
        _write(work_dir, "tests/test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(30)\n")

        started = time.time()
        result = RealKnight(timeout=1).verify(_context(work_dir), _plan())

        assert result.success is False
        assert result.verification_results[0]["status"] == "timeout"
        assert time.time() - started < 25

    def test_execute_never_claims_a_write_it_did_not_make(self, work_dir):
        plan = _plan(
            operations=[
                {"type": "add_null_check", "file": "api/handler.py", "line": 45, "code": "if x is None:\n    pass"}
            ]
        )

        result = RealKnight().execute(_context(work_dir), plan)

        assert result.success is False
        assert result.artifacts["applied_files"] == []
        assert not (work_dir / "api" / "handler.py").exists()
        assert result.verification_results[0]["status"] in {"failed", "refused", "skipped"}

    def test_execute_without_operations_says_so_instead_of_succeeding(self, work_dir):
        result = RealKnight().execute(_context(work_dir), _plan(operations=[]))

        assert result.success is False
        assert "没有可执行的操作" in result.message


class TestPluginModes:
    def test_real_mode_is_the_default(self):
        plugin = DefaultAutotestGenPlugin()

        assert plugin.mode == "real"
        assert isinstance(plugin.get_perceiver(), RealPerceiver)
        assert isinstance(plugin.get_knight(), RealKnight)

    def test_plugin_manager_loads_the_real_mode(self):
        plugin = PluginManager().get_plugin("default")

        assert isinstance(plugin.get_perceiver(), RealPerceiver)
        assert isinstance(plugin.get_knight(), RealKnight)

    def test_stub_mode_keeps_the_old_demo_behaviour_for_comparison(self):
        plugin = DefaultAutotestGenPlugin(mode="stub")
        context = plugin.create_context(".")

        pain_points = plugin.get_perceiver().detect(context)

        assert isinstance(plugin.get_perceiver(), StubPerceiver)
        assert isinstance(plugin.get_knight(), StubKnight)
        # 旧行为的全部证据：位置是写死的，成功是无条件的
        assert sorted({point.location for point in pain_points}) == ["api/handler.py", "utils/processor.py"]
        plan = plugin.get_sage().generate_patch(context, pain_points[0])
        assert plugin.get_knight().verify(context, plan).success is True

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="未知 mode"):
            DefaultAutotestGenPlugin(mode="guess")


def _plan(operations=None) -> PatchPlan:
    return PatchPlan(
        id="patch-1",
        pain_point_id="pain-1",
        description="test plan",
        operations=[{"type": "noop", "file": "mod.py"}] if operations is None else operations,
        estimated_effort=0.0,
    )
