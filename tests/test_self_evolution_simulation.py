"""自进化联邦学习入口的「模拟」声明测试。

``SelfEvolvingFederatedLearning.run_single_experiment`` 不做真实联邦训练：它用合成任务
加带噪声的预测值构造「实验结果」。原来的 CLI 只把这件事写在注释里
（``# Simulate experiment (in real system, this would run actual FL)``），输出里一个字都
没有，于是 0.91 的「Actual Accuracy」很容易被读成真实联邦训练的成绩。

所以这里断言的是**声明必须在证据之前**：入口在打印任何性能数字之前，先打印
``[NOTE] 本模块为模拟环境，非真实联邦训练``；模块与 ``run_single_experiment`` 的
docstring 也都要写明这一点。
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes_unified.self_evolution import run_self_evolution as runner  # noqa: E402  (须先补 sys.path)

SIMULATION_NOTICE = "[NOTE] 本模块为模拟环境，非真实联邦训练"


class _StubSelfEvolvingFL:
    """替身：只记录有没有被调用，不跑那几十秒的预测器训练。"""

    def __init__(self):
        self.evolved_with = None

    def evolve(self, num_generations, max_time_hours):
        self.evolved_with = (num_generations, max_time_hours)

    def generate_evolution_report(self):
        return {
            "summary": {
                "total_experiments": 0,
                "generations": 0,
                "best_accuracy": 0.0,
                "milestones": 0,
            },
            "algorithm_statistics": {},
            "best_configuration": None,
        }


class TestSimulationNotice:
    def test_cli_prints_the_simulation_notice(self, monkeypatch, capsys):
        monkeypatch.setattr(runner, "SelfEvolvingFederatedLearning", _StubSelfEvolvingFL)

        runner.run_self_evolution(num_generations=1, max_time_hours=0.0, start_server=False)

        out = capsys.readouterr().out
        assert SIMULATION_NOTICE in out

    def test_notice_comes_before_any_performance_number(self, monkeypatch, capsys):
        monkeypatch.setattr(runner, "SelfEvolvingFederatedLearning", _StubSelfEvolvingFL)

        runner.run_self_evolution(num_generations=1, max_time_hours=0.0, start_server=False)

        out = capsys.readouterr().out
        assert out.index(SIMULATION_NOTICE) < out.index("Total Experiments:")
        assert out.index(SIMULATION_NOTICE) < out.index("Best Accuracy:")

    def test_module_docstring_declares_simulation(self):
        doc = runner.__doc__ or ""

        assert "[NOTE]" in doc
        assert "模拟" in doc
        assert "不是真实联邦训练" in doc

    def test_run_single_experiment_docstring_declares_simulation(self):
        from hermes_unified.self_evolution.self_evolution import (
            SelfEvolvingFederatedLearning,
        )

        doc = SelfEvolvingFederatedLearning.run_single_experiment.__doc__ or ""

        assert "[NOTE]" in doc
        assert "模拟" in doc

    def test_no_server_run_still_reports_the_notice(self, monkeypatch, capsys):
        """``--no-server`` 也要看到声明：它不该只出现在 dashboard 路径里。"""

        monkeypatch.setattr(runner, "SelfEvolvingFederatedLearning", _StubSelfEvolvingFL)

        runner.run_self_evolution(num_generations=1, max_time_hours=0.0, start_server=False)

        assert SIMULATION_NOTICE in capsys.readouterr().out

    def test_evolution_is_actually_driven_with_the_given_budget(self, monkeypatch):
        stub = _StubSelfEvolvingFL()
        monkeypatch.setattr(runner, "SelfEvolvingFederatedLearning", lambda: stub)

        runner.run_self_evolution(num_generations=7, max_time_hours=0.25, start_server=False)

        assert stub.evolved_with == (7, 0.25)
