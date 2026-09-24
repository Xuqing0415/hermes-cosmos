"""unused_import_miner 的单元测试。全部用真实临时 git 仓库，配对规则真的跑了一遍 git。"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import uuid

import pytest

from hermes.self_research.unused_import_miner import (
    REDEFINITION,
    UNUSED,
    WEAK_DOTTED_SIDE_EFFECT,
    WEAK_DYNAMIC_USAGE,
    WEAK_PACKAGE_INIT,
    WEAK_REDEFINITION,
    WEAK_STAR_IMPORT,
    WEAK_VERSION_BRANCH,
    dunder_all_names,
    mine_repository,
    unused_import_names,
    version_branch_imports,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GIT_ENV = {
    "GIT_AUTHOR_NAME": "unused",
    "GIT_AUTHOR_EMAIL": "unused@example.com",
    "GIT_COMMITTER_NAME": "unused",
    "GIT_COMMITTER_EMAIL": "unused@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


@pytest.fixture
def work_dir():
    base = os.path.join(REPO_ROOT, ".tmp_test", "unused_import_miner")
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


def _repo(root: pathlib.Path) -> pathlib.Path:
    repo = root / "project"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    return repo


def _write(repo: pathlib.Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: pathlib.Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, "--no-gpg-sign")


def _cases(repo: pathlib.Path) -> list:
    return mine_repository(str(repo), limit=None, max_cases=50).cases


class TestPyflakesExtraction:
    def test_reports_line_and_raw_name(self):
        # pyflakes 的 str(message) 带 文件:行:列: 前缀；用 ^...$ 锚点会一条都抓不到
        assert unused_import_names("import os\n", "m.py") == [(1, "os", UNUSED)]

    def test_from_import_is_reported_as_module_dot_name(self):
        assert unused_import_names("from typing import Dict\n", "m.py") == [(1, "typing.Dict", UNUSED)]

    def test_used_import_is_not_reported(self):
        assert unused_import_names("import os\n\n\ndef f():\n    return os\n", "m.py") == []

    def test_redefinition_is_its_own_kind(self):
        source = "import os\nimport os\n\n\ndef f():\n    return os\n"
        assert unused_import_names(source, "m.py") == [(2, "os", REDEFINITION)]

    def test_literal_dunder_all_counts_as_use(self):
        assert unused_import_names("import os\n\n__all__ = ['os']\n", "m.py") == []


class TestDunderAll:
    def test_literal_list(self):
        assert dunder_all_names("__all__ = ['a', 'b']\n") == {"a", "b"}

    def test_augmented_assignment(self):
        assert dunder_all_names("__all__ = []\n__all__ += ['c']\n") == {"c"}

    def test_dynamic_all_is_invisible(self):
        # 拼出来的 __all__ pyflakes 看不见 —— 这正是 in_dunder_all 这个 weak 标签存在的理由
        assert dunder_all_names("__all__ = BASE + ['x']\n") == {"x"}

    def test_unparsable_source_yields_nothing(self):
        assert dunder_all_names("def (:\n") == set()


class TestVersionBranch:
    def test_sys_version_info_conditional(self):
        source = "import sys\n\nif sys.version_info < (3,):\n    import Queue as queue\nelse:\n    import queue\n"
        assert version_branch_imports(source) == {"queue"}

    def test_flag_variable_conditional(self):
        source = "PY2 = sys.version_info[0] == 2\n\nif PY2:\n    import Queue\nelse:\n    import queue\n"
        assert version_branch_imports(source) == {"Queue", "queue"}

    def test_flag_derived_from_another_flag_is_recognized(self):
        # requests/flask 的真实写法：先把 sys.version_info 存进 _ver，再由 _ver 派生 is_py2。
        # 只认第一手会整个漏掉这一类 —— 而这些项目的分支变量几乎全是第二手。
        source = (
            "_ver = sys.version_info\n\n"
            "is_py2 = (_ver[0] == 2)\n\n"
            "if is_py2:\n    from urlparse import urlparse\nelse:\n    from urllib.parse import urlparse\n"
        )
        assert version_branch_imports(source) == {"urlparse"}

    def test_platform_flags_are_not_version_flags(self):
        # 同一段代码里往往混着平台判断，它们跟版本无关，不能顺着同一个闭包被吸进来
        source = (
            "_ver = sys.version_info\n\n"
            "is_py2 = (_ver[0] == 2)\n\n"
            "is_windows = 'win32' in str(sys.platform).lower()\n\n"
            "if is_windows:\n    import msvcrt\n"
        )
        assert version_branch_imports(source) == set()

    def test_nested_import_inside_a_branch_is_found(self):
        source = "if sys.version_info >= (3,):\n    if True:\n        import json\n"
        assert version_branch_imports(source) == {"json"}

    def test_plain_conditional_is_not_version_gated(self):
        assert version_branch_imports("if DEBUG:\n    import json\n") == set()

    def test_unparsable_source_yields_nothing(self):
        assert version_branch_imports("def (:\n") == set()


class TestMiner:
    def test_pairs_unused_import_with_its_deletion(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os\nimport json\n\n\ndef go():\n    return json.dumps({})\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import json\n\n\ndef go():\n    return json.dumps({})\n")
        _commit(repo, "cleanup: drop unused os")

        cases = _cases(repo)
        assert len(cases) == 1
        assert (cases[0].file, cases[0].line, cases[0].names) == ("app.py", 1, ["os"])
        assert cases[0].is_clean

    def test_adding_an_import_is_not_counted_as_removing_one(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import json\n\n\ndef go():\n    return json.dumps({})\n")
        _commit(repo, "feat: use json")

        assert _cases(repo) == []

    def test_deleting_a_still_used_import_is_not_a_pair(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "refactor: drop os")

        assert _cases(repo) == []

    def test_one_statement_binding_many_names_is_one_case(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "from urllib.parse import urljoin, urlparse\n\n\ndef go():\n    return 1\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "cleanup: drop urllib")

        cases = _cases(repo)
        assert len(cases) == 1
        assert cases[0].names == ["urljoin", "urlparse"]
        assert cases[0].statement == "from urllib.parse import urljoin, urlparse"

    def test_package_init_deletion_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "pkg/__init__.py", "from . import helpers\n")
        _write(repo, "pkg/helpers.py", "VALUE = 1\n")
        _commit(repo, "feat: pkg")
        _write(repo, "pkg/__init__.py", "")
        _commit(repo, "cleanup: drop import")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_PACKAGE_INIT in cases[0].weak
        assert not cases[0].is_clean

    def test_star_import_in_the_file_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "from os import *\nimport json\n\n\ndef go():\n    return 1\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "from os import *\n\n\ndef go():\n    return 1\n")
        _commit(repo, "cleanup: drop json")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_STAR_IMPORT in cases[0].weak

    def test_dynamic_lookup_in_the_file_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import json\n\n\ndef go(obj):\n    return getattr(obj, 'x')\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "def go(obj):\n    return getattr(obj, 'x')\n")
        _commit(repo, "cleanup: drop json")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_DYNAMIC_USAGE in cases[0].weak

    def test_dotted_import_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os.path\n\n\ndef go():\n    return 1\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "cleanup: drop os.path")

        cases = _cases(repo)
        assert len(cases) == 1
        assert cases[0].names == ["os"]
        assert WEAK_DOTTED_SIDE_EFFECT in cases[0].weak

    def test_redefinition_deletion_is_weak(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os\nimport os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "feat: go")
        _write(repo, "app.py", "import os\n\n\ndef go():\n    return os.getcwd()\n")
        _commit(repo, "cleanup: drop duplicate import")

        cases = _cases(repo)
        assert len(cases) == 1
        assert WEAK_REDEFINITION in cases[0].weak

    def test_dunder_all_export_is_not_mined(self, work_dir):
        repo = _repo(work_dir)
        _write(repo, "app.py", "import os\n\n__all__ = ['os']\n")
        _commit(repo, "feat: app")
        _write(repo, "app.py", "__all__ = []\n")
        _commit(repo, "refactor: drop the export")

        assert _cases(repo) == []

    def test_one_name_imported_in_both_version_branches_is_ambiguous(self, work_dir):
        # 同名导入出现在版本判断的两个分支里 —— 删除是安全的，但「该删哪一支」要看项目的
        # 目标版本，静态判不了，所以单列成 ambiguous 而不是当成「同一个名字出现两次」。
        repo = _repo(work_dir)
        _write(
            repo,
            "app.py",
            "import sys\n\nif sys.version_info < (3,):\n    import Queue as queue\nelse:\n"
            "    import queue\n\n\ndef go():\n    return 1\n",
        )
        _commit(repo, "feat: queue shim")
        _write(repo, "app.py", "def go():\n    return 1\n")
        _commit(repo, "cleanup: drop the queue shim")

        cases = _cases(repo)
        assert cases, "这条配对应该被挖出来"
        assert all(WEAK_VERSION_BRANCH in case.weak for case in cases)
        assert not any(case.is_clean for case in cases)
