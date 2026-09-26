"""跨域迁移那一行的三个数字，必须说的是同一件事。

原来打印的是 ``{迁移次数}/{全部用例数} ({整体通过率}%)``：分子是迁移次数、分母是全部
用例、括号里却是另一个比例。屏幕上于是会出现 ``0/15 (73%)`` —— 读起来像「15 次迁移
一次都没成，但整体通过 73%」。三个数字互不相干，就是不自洽。

断言走子进程：``import autotestgen`` 会在导入时把 sys.stdout/stderr 换成 UTF-8 包装
（Windows 上），在 pytest 进程里 import 会把捕获用的流也一起卷进去（这正是批次 4.4 要修
的那个副作用）。这里不跟它纠缠，直接开一个干净的解释器问它要那一行。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINE = re.compile(r"Cross-domain transfer: (\d+)/(\d+) cases \((\d+)%\)")

PROBE = (
    "import json, sys;"
    "sys.path.insert(0, '.');"
    "import autotestgen;"
    "print(json.dumps([autotestgen.format_cross_domain_line(t, n) for t, n in json.loads(sys.argv[1])]))"
)


def _lines(pairs):
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(pairs)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


class TestCrossDomainLine:
    def test_numerator_denominator_and_percentage_agree(self):
        for (transfers, total), line in zip(((0, 15), (3, 12), (7, 7)), _lines([(0, 15), (3, 12), (7, 7)])):
            match = LINE.search(line)
            assert match, f"这一行必须能被机器读回来，否则没法断言自洽：{line}"
            got_transfers, got_total, percent = (int(group) for group in match.groups())
            assert (got_transfers, got_total) == (transfers, total)
            assert percent == round(transfers / total * 100)

    def test_zero_transfers_is_zero_percent(self):
        """真实跑出来的那一行：0/15。以前括号里写的是整体通过率 73%。"""

        assert _lines([(0, 15)])[0].endswith("0/15 cases (0%)")

    def test_no_cases_does_not_divide_by_zero(self):
        assert _lines([(0, 0)])[0].endswith("0/0 cases (0%)")
