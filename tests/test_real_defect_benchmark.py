"""真实缺陷基准的单元测试与端到端测试。

测试里用的是一个**真实 git 仓库**（临时创建、含真实缺陷与真实测试），
因此不依赖网络：基准的每一环（采样、复现、对照组、系统自称 vs 实际）都真的跑了一遍。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import uuid

import pytest

from hermes.self_research.real_defect_benchmark import (
    DefectCase,
    DefectSampler,
    GitError,
    GitRepo,
    RealDefectBenchmark,
    _extract_count,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GIT_ENV = {
    "GIT_AUTHOR_NAME": "bench",
    "GIT_AUTHOR_EMAIL": "bench@example.com",
    "GIT_COMMITTER_NAME": "bench",
    "GIT_COMMITTER_EMAIL": "bench@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


@pytest.fixture
def work_dir():
    base = os.path.join(REPO_ROOT, ".tmp_test", "real_defect_benchmark")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _git(repo: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **GIT_ENV},
    )
    assert completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr}"
    return completed.stdout


def _commit(repo: pathlib.Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, "--no-gpg-sign")
    return _git(repo, "rev-parse", "HEAD").strip()


def _write(repo: pathlib.Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _buggy_then_fixed_repo(root: pathlib.Path) -> pathlib.Path:
    """c0 正常 -> c1 引入缺陷（无测试）-> c2 修复并补测试。"""

    repo = root / "project"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "calc.py", "def add(a, b):\n    return a + b\n")
    _write(repo, "tests/test_calc.py", "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    _commit(repo, "init: 加法")
    _write(repo, "calc.py", "def add(a, b):\n    return a + b\n\n\ndef div(a, b):\n    return a + b\n")
    _commit(repo, "feat: 添加除法")
    _write(repo, "calc.py", "def add(a, b):\n    return a + b\n\n\ndef div(a, b):\n    return a / b\n")
    _write(
        repo,
        "tests/test_calc_div.py",
        "from calc import div\n\n\ndef test_div():\n    assert div(6, 3) == 2\n",
    )
    _commit(repo, "fix: 修正 div 的实现")
    return repo


class TestDefectSampler:
    def test_picks_the_fix_commit_with_source_and_tests(self, work_dir):
        repo = GitRepo(str(_buggy_then_fixed_repo(work_dir)))

        cases = DefectSampler(repo).sample(5)

        assert len(cases) == 1
        case = cases[0]
        assert case.subject.startswith("fix:")
        assert case.source_files == ["calc.py"]
        assert case.test_files == ["tests/test_calc_div.py"]
        assert case.parent != case.sha

    def test_skips_fix_commits_without_tests(self, work_dir):
        repo_path = work_dir / "notests"
        repo_path.mkdir()
        _git(repo_path, "init", "-b", "main")
        _write(repo_path, "calc.py", "VALUE = 1\n")
        _commit(repo_path, "init")
        _write(repo_path, "calc.py", "VALUE = 2\n")
        _commit(repo_path, "fix: 调个常量")

        cases = DefectSampler(GitRepo(str(repo_path))).sample(5)

        assert cases == []

    def test_rejects_a_directory_that_is_not_a_repository(self, work_dir):
        plain = work_dir / "plain"
        plain.mkdir()

        with pytest.raises(GitError, match="不是 git 仓库"):
            GitRepo(str(plain))


class TestRealDefectBenchmark:
    def test_real_defect_is_reproduced_confirmed_and_then_not_repaired(self, work_dir):
        repo = _buggy_then_fixed_repo(work_dir)

        report = RealDefectBenchmark(str(repo), str(work_dir / "wt"), cases=1).run()

        assert report.total == 1
        result = report.cases[0]
        # 父提交上失败、修复提交上通过：这个缺陷与这个测试都是真的
        assert result.reproduced is True
        assert result.confirmed is True
        assert result.inconclusive is False
        # 真实感知器按证据说话：这个仓库里没有它能证明的静态问题，于是它什么都不报。
        # 不报比乱报强——旧假桩正是在这里报出写死的 api/handler.py。
        assert result.claimed_locations == []
        assert result.claimed_pain_points == 0
        assert result.files_touched == []
        assert result.detected is False
        # 没有操作可执行、也没有测试通过：系统不再自称成功，于是也没有虚假声称
        assert result.claimed_success is False
        assert result.repaired is False
        assert result.false_claim is False

        payload = report.to_dict()
        assert payload["conclusive"] == 1
        assert payload["detection_rate"] == 0.0
        assert payload["repair_rate"] == 0.0
        assert payload["false_claim_rate"] == 0.0
        assert report.summary_line().startswith("真实缺陷 1 例")

    def test_benchmark_measures_the_real_plugin_not_the_stub(self, work_dir):
        """基准必须测真实实现：假桩才会报 api/handler.py 并自称成功。"""

        repo = _buggy_then_fixed_repo(work_dir)
        benchmark = RealDefectBenchmark(str(repo), str(work_dir / "wt6"), cases=1)

        claims = benchmark._run_system(str(repo))

        assert claims["locations"] == []
        assert claims["success"] is False

    def test_control_failure_marks_the_case_inconclusive_and_excludes_it(self, work_dir):
        repo = work_dir / "always_red"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _write(repo, "calc.py", "def div(a, b):\n    return a + b\n")
        _commit(repo, "init")
        # 这个测试在修复提交上也会失败（断言本身不成立），因此不能用来评价系统
        _write(repo, "calc.py", "def div(a, b):\n    return a / b\n")
        _write(repo, "tests/test_always_red.py", "def test_never_true():\n    assert False\n")
        _commit(repo, "fix: 修正 div 的实现")

        report = RealDefectBenchmark(str(repo), str(work_dir / "wt2"), cases=1).run()

        result = report.cases[0]
        assert result.reproduced is True
        assert result.confirmed is False
        assert result.inconclusive is True
        assert result.false_claim is False
        assert report.inconclusive == 1
        assert report.to_dict()["detection_rate"] == 0.0
        assert any("对照组" in note for note in result.notes)

    def test_no_cases_means_no_conclusions(self, work_dir):
        repo = work_dir / "clean"
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _write(repo, "calc.py", "VALUE = 1\n")
        _commit(repo, "init: 只有正常提交")

        report = RealDefectBenchmark(str(repo), str(work_dir / "wt3"), cases=3).run()

        assert report.total == 0
        assert report.to_dict()["detection_rate"] == 0.0
        assert report.to_dict()["repair_rate"] == 0.0
        assert any("没有采到" in note for note in report.notes)

    def test_non_repository_is_reported_not_guessed(self, work_dir):
        plain = work_dir / "plain"
        plain.mkdir()

        report = RealDefectBenchmark(str(plain), str(work_dir / "wt4"), cases=1).run()

        assert report.git_available is False
        assert report.total == 0
        assert any("不做任何推断" in note for note in report.notes)

    def test_markdown_report_lists_every_case(self, work_dir):
        repo = _buggy_then_fixed_repo(work_dir)

        report = RealDefectBenchmark(str(repo), str(work_dir / "wt5"), cases=1).run()
        markdown = report.markdown()

        assert "## 真实缺陷基准" in markdown
        assert "| 用例 | 修复提交 |" in markdown
        assert report.cases[0].case.sha[:8] in markdown
        assert "真正修好" in markdown


class TestHelpers:
    def test_detection_requires_the_actual_file(self):
        case = DefectCase(
            sha="a" * 40,
            parent="b" * 40,
            subject="fix: x",
            date="2026-01-01",
            source_files=["src/click/core.py"],
            test_files=["tests/test_core.py"],
        )

        assert RealDefectBenchmark._detected(case, ["src/click/core.py"], []) is True
        assert RealDefectBenchmark._detected(case, ["./src/click/core.py"], []) is True
        assert RealDefectBenchmark._detected(case, ["api/handler.py"], []) is False
        assert RealDefectBenchmark._detected(case, [], ["src/click/core.py"]) is True
        assert RealDefectBenchmark._detected(case, [], []) is False

    def test_pytest_counts_are_parsed_from_the_summary_line(self):
        output = "FAILED tests/test_x.py::test_y\n1 failed, 798 passed, 2 errors in 1.99s"

        assert _extract_count(output, "passed") == 798
        assert _extract_count(output, "failed") == 1
        assert _extract_count(output, "error") == 2
        assert _extract_count("no summary here", "passed") == 0
