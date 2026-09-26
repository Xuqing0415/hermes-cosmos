"""Windows 控制台（GBK）不该因为一个符号就崩，也不该被 `import` 悄悄改掉。

两条断言：

1. ``hermes_unified/`` 里不存在 GBK（cp936）打不出来的字符。少一个 emoji，就少一条
   只在中文 Windows 上复现的 ``UnicodeEncodeError``——它跟终端宽度、字体都无关，只跟
   「这个字符能不能编码」有关，所以可以用编码本身来判定。
2. ``import autotestgen`` 不再替换 ``sys.stdout``/``stderr``。以前这一步发生在导入时，
   把 pytest 的 capsys 一起卷走了，于是同一个测试时好时坏——取决于谁先导入。
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = pathlib.Path(REPO_ROOT) / "hermes_unified"


def _import_autotestgen():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import autotestgen

    return autotestgen


def _unencodable_characters(text):
    for character in text:
        try:
            character.encode("cp936")
        except UnicodeEncodeError:
            yield character


class TestSourceIsGbkSafe:
    def test_no_character_in_hermes_unified_is_unencodable_in_gbk(self):
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for character in _unencodable_characters(line):
                    offenders.append(
                        "{}:{}: {!r} (U+{:04X})".format(path.relative_to(REPO_ROOT), number, character, ord(character))
                    )
                    break

        assert not offenders, "这些字符在 GBK 控制台上会抛 UnicodeEncodeError：\n" + "\n".join(offenders)


class TestAutotestgenImportHasNoConsoleSideEffect:
    def test_reimporting_autotestgen_leaves_the_console_streams_alone(self):
        """reload 会重新执行模块顶层代码，等价于「第一次导入」那一下。"""

        module = _import_autotestgen()
        stdout_before, stderr_before = sys.stdout, sys.stderr

        importlib.reload(module)

        assert sys.stdout is stdout_before
        assert sys.stderr is stderr_before

    def test_main_restores_the_console_streams_when_it_exits(self, monkeypatch, capsys):
        module = _import_autotestgen()
        stdout_before, stderr_before = sys.stdout, sys.stderr
        monkeypatch.setattr(sys, "argv", ["autotestgen.py"])

        with pytest.raises(SystemExit):
            module.main()

        assert sys.stdout is stdout_before
        assert sys.stderr is stderr_before
        assert "--domain is required" in capsys.readouterr().out
