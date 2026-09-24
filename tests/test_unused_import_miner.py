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
    dunder_all_names,
    mine_repository,
    unused_import_names,
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
