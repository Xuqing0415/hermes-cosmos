"""import_fix_miner / import_fix_benchmark 的单元与端到端测试。

测试用的是**真实 git 仓库**（临时创建），所以「挖配对」这件事真的跑了一遍 git，
不是拿假的 diff 字符串糊过去。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import uuid

import pytest

from hermes.self_research import import_fix_benchmark as bench
from hermes.self_research.import_fix_miner import (
    WEAK_CROSS_FILE,
    WEAK_NOT_RESOLVED,
    WEAK_PY2,
    WEAK_TYPE_CHECKING,
    import_bindings,
    mine_repository,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GIT_ENV = {
    "GIT_AUTHOR_NAME": "miner",
    "GIT_AUTHOR_EMAIL": "miner@example.com",
    "GIT_COMMITTER_NAME": "miner",
    "GIT_COMMITTER_EMAIL": "miner@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


@pytest.fixture
def work_dir():
    base = os.path.join(REPO_ROOT, ".tmp_test", "import_fix_miner")
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


def _write(repo: pathlib.Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: pathlib.Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, "--no-gpg-sign")


def _repo(root: pathlib.Path) -> pathlib.Path:
    repo = root / "project"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    return repo


def _cases(repo: pathlib.Path) -> list:
    return mine_repository(str(repo), limit=None, max_cases=50).cases


class TestImportBindings:
    def test_plain_import_binds_top_level_module(self):
        assert import_bindings("import os.path") == {"os": ("", "os.path")}

    def test_from_import_binds_each_name(self):
        assert import_bindings("from typing import List, Dict") == {
            "List": ("typing", "List"),
            "Dict": ("typing", "Dict"),
        }

    def test_relative_import_keeps_module_tail(self):
        assert import_bindings("from . import helpers") == {"helpers": ("", "helpers")}

    def test_aliased_import_binds_the_alias(self):
        assert import_bindings("import numpy as np") == {"np": ("", "numpy")}

    def test_non_import_statement_binds_nothing(self):
        assert import_bindings("x = 1") == {}


class TestMiner:
    def test_pairs_parent_undefined_name_with_added_import(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return os.getcwd()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "fix: missing import")

        cases = _cases(repo)
        assert len(cases) == 1
        case = cases[0]
        assert (case.file, case.name) == ("app.py", "os")
        assert case.human_statement == "import os"
        assert case.human_key == ("", "os")
        assert case.resolved is True
        assert case.is_clean

    def test_deleting_an_import_is_not_counted_as_adding_one(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n\n\ndef unused():\n    return 1\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "cleanup: drop imports")

        assert _cases(repo) == []

    def test_repeated_use_of_one_name_is_one_case(self, work_dir):
        repo = _repo(work_dir)
        # 两个使用点必须在不同行上：pyflakes 按 (行, 名字) 报，"按行去重"就是这里的语义
        _write(repo, "app.py", "def go():\n    here = os.getcwd()\n    return here + os.sep\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import os\n\n\ndef go():\n    here = os.getcwd()\n    return here + os.sep\n")
        _commit(repo, "fix: missing import")

        cases = _cases(repo)
        assert len(cases) == 1
        assert cases[0].use_sites == 2

    def test_type_checking_import_is_weak_not_clean(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return typing_extensions.override\n")
        _commit(repo, "feat: go")
        _write(
            repo,
            "app.py",
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
            "    import typing_extensions\n\n\ndef go():\n    return 1\n",
        )
        _commit(repo, "fix: guard the import")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_TYPE_CHECKING in cases[0].weak
        assert not cases[0].is_clean

    def test_python2_only_builtin_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go(value):\n    return isinstance(value, basestring)\n")
        _commit(repo, "feat: go")
        _write(
            repo,
            "app.py",
            "from builtins import basestring\n\n\ndef go(value):\n    return isinstance(value, basestring)\n",
        )
        _commit(repo, "fix: py2 leftover")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_PY2 in cases[0].weak

    def test_import_in_another_file_does_not_count_as_resolved(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "a.py", "def go():\n    return os.getcwd()\n")
        _write(repo, "b.py", "def go():\n    return os.sep\n")
        _commit(repo, "feat: two users of os")
        # a.py 必须真的出现在这个提交的 diff 里，否则它根本不参与配对，测不到跨文件那条规则
        _write(repo, "a.py", "# still broken\n\n\ndef go():\n    return os.getcwd()\n")
        _write(repo, "b.py", "import os\n\n\ndef go():\n    return os.sep\n")
        _commit(repo, "fix: import os in b only")

        cases = {(case.file, case.name): case for case in _cases(repo)}
        assert ("b.py", "os") in cases and cases[("b.py", "os")].is_clean
        assert ("a.py", "os") in cases
        assert WEAK_NOT_RESOLVED in cases[("a.py", "os")].weak
        assert WEAK_CROSS_FILE in cases[("a.py", "os")].weak


class TestAgreement:
    def test_exact_match(self):
        assert bench._agreement(("", "os"), ("", "os")) == bench.AGREE_EXACT

    def test_relative_and_absolute_module_are_loosely_equal(self):
        assert bench._agreement(("flask._compat", "reraise"), ("_compat", "reraise")) == bench.AGREE_LOOSE

    def test_different_module_is_disagreement(self):
        assert bench._agreement(("werkzeug.utils", "os"), ("", "os")) == bench.AGREE_DIFFERENT

    def test_another_level_of_the_same_package_is_its_own_bucket(self):
        assert bench._agreement(("flask", "current_app"), ("flask.globals", "current_app")) == bench.AGREE_SAME_PACKAGE
        assert bench._agreement(("importlib", "importlib"), ("importlib.util", "importlib")) == bench.AGREE_SAME_PACKAGE

    def test_whole_module_versus_symbol_from_a_module_is_not_loose(self):
        # 空模块名会让 endswith 变成万能匹配，必须挡住，否则一致率虚高
        assert bench._agreement(("", "json"), ("helpers", "json")) == bench.AGREE_DIFFERENT

    def test_mismatched_name_is_disagreement(self):
        assert bench._agreement(("flask", "request"), ("flask.globals", "session")) == bench.AGREE_DIFFERENT

    def test_unbound_system_statement_is_disagreement(self):
        assert bench._agreement(None, ("", "os")) == bench.AGREE_DIFFERENT


class TestSplice:
    def test_inserts_after_the_given_line(self):
        assert bench._splice("a\nb\n", 1, "import os") == "a\nimport os\nb\n"

    def test_zero_inserts_at_the_top(self):
        assert bench._splice("a\n", 0, "import os") == "import os\na\n"


class TestEvaluateRepository:
    def test_work_root_is_resolved_against_the_working_directory_not_the_repo(self, work_dir):
        # git -C <repo> worktree add <相对路径> 是相对那个仓库解析的。传相对路径曾让 checkout
        # 落到别处、而 harness 拿到一个空目录，于是检出率静默变成 0.00 —— 假报告。
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return os.getcwd()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "fix: missing import")

        relative = os.path.relpath(os.path.join(str(work_dir), "trees"))
        report = bench.evaluate_repository(str(repo), work_root=relative, max_pairs=10)

        assert report.detection_rate == 1.0
        assert report.outcomes[0].decision == bench.DECISION_PATCH

    def test_a_tree_without_python_files_is_refused_not_scored(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "README.md", "no python here\n")
        _commit(repo, "docs")

        trees = bench.ParentTree(bench.GitRepo(str(repo)), os.path.join(str(work_dir), "trees"))
        assert trees.open("HEAD") is None

    def test_oracle_agrees_with_the_human_on_a_stdlib_name(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return os.getcwd()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "fix: missing import")

        report = bench.evaluate_repository(str(repo), work_root=os.path.join(str(work_dir), "trees"), max_pairs=10)

        assert report.pairs == 1
        outcome = report.outcomes[0]
        assert outcome.detected is True
        assert outcome.decision == bench.DECISION_PATCH
        assert outcome.system_statement == "import os"
        assert outcome.agreement == bench.AGREE_EXACT
        assert outcome.false_positive is False
        assert report.detection_rate == 1.0
        assert report.agree_rate == 1.0
        assert report.abstain_rate == 0.0
        assert report.false_positive_rate == 0.0

    def test_oracle_abstains_when_the_name_has_no_provable_source(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return helper()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "from private_helpers import helper\n\n\ndef go():\n    return helper()\n")
        _commit(repo, "fix: missing import")

        report = bench.evaluate_repository(str(repo), work_root=os.path.join(str(work_dir), "trees"), max_pairs=10)

        assert report.pairs == 1
        assert report.outcomes[0].decision == bench.DECISION_ABSTAIN
        assert report.abstain_rate == 1.0
        assert report.agree_rate is None
