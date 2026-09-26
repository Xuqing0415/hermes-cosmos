"""攻击进化 GA 报出来的「最佳个体」，必须就是那个适应度所对应的个体。

原来的写法先把 ``self.population`` 换成 ``new_population``，再拿**旧种群**的
``fitnesses`` 去 ``argmax`` 索引**新种群** —— 屏幕上出现的强度和适应度不是同一条
记录。这里用「适应度 = intensity / 10」的桩模拟器把两者钉死：报出来的个体只要和
适应度对不上，断言就会失败。

同一个文件还看着 ``AttackDefenseGame``：它每轮都算出了攻击参数，但以前不往外传，
于是 ``run_game`` 那句 ``intensity=...`` 永远打印 0.00。
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes_unified.federated.adaptive_defense import AttackDefenseGame  # noqa: E402
from hermes_unified.federated.attack_evolution import AttackEvolution  # noqa: E402

BEST_FITNESS = re.compile(r"^\s*Best Fitness: ([\d.]+)")
BEST_INTENSITY = re.compile(r"^\s*- Intensity: ([\d.]+)")
FINAL_FITNESS = re.compile(r"^Best Attack Fitness: ([\d.]+)")
FINAL_PARAMS = re.compile(r"^Best Attack Parameters: (.*)$")
PARAM_INTENSITY = re.compile(r"'intensity': ([\d.]+)")


class _IntensitySimulator:
    """适应度完全由 intensity 决定：fitness = intensity / 10，可手算复核。"""

    def __init__(self, attack_config=None, **_ignored):
        self.attack_config = attack_config or {}

    def run_federated_training(self, **_ignored):
        return {"accuracy_history": [1.0 - self.attack_config["intensity"] / 10.0]}


def _individual(intensity, attack_type="gradient_scale"):
    return {
        "attack_type": attack_type,
        "intensity": intensity,
        "mal_ratio": 0.3,
        "start_round": 10,
        "trigger_pos": None,
    }


def _evolve_and_capture(monkeypatch, generations=3):
    """跑一轮进化，把 stdout 收回来。父代固定选择「最差」个体，好让错位必然暴露。"""

    monkeypatch.setattr(random, "choices", lambda population, weights=None, k=1: [0] * k)
    monkeypatch.setattr(AttackEvolution, "_crossover", lambda self, p1, p2: dict(p1))
    monkeypatch.setattr(AttackEvolution, "_mutate", lambda self, ind: ind)

    evolution = AttackEvolution(_IntensitySimulator, {}, population_size=4, generations=generations)
    evolution.population = [_individual(intensity) for intensity in (1.0, 4.0, 7.0, 9.0)]

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        evolution.evolve()

    return stdout.getvalue()


def _pairs_from_output(text):
    """抽出 (适应度, 该适应度对应的个体强度) —— 两处打印都要看着自洽。"""

    pairs = []
    pending = None
    for line in text.splitlines():
        match = BEST_FITNESS.match(line) or FINAL_FITNESS.match(line)
        if match:
            pending = float(match.group(1))
            continue
        match = BEST_INTENSITY.match(line)
        if match and pending is not None:
            pairs.append((pending, float(match.group(1))))
            pending = None
            continue
        match = FINAL_PARAMS.match(line)
        if match and pending is not None:
            intensity = PARAM_INTENSITY.search(match.group(1))
            assert intensity, f"最佳个体里应该有 intensity：{line}"
            pairs.append((pending, float(intensity.group(1))))
            pending = None

    return pairs


class TestReportedBestIndividual:
    def test_every_reported_fitness_belongs_to_the_reported_individual(self, monkeypatch):
        pairs = _pairs_from_output(_evolve_and_capture(monkeypatch))

        assert len(pairs) == 4, f"3 代各一次 + 结尾一次，实际收到 {len(pairs)} 次"
        for fitness, intensity in pairs:
            assert fitness == intensity / 10.0, f"适应度 {fitness} 不属于强度 {intensity} 的个体"

    def test_first_generation_reports_the_true_best_of_that_population(self, monkeypatch):
        """初始种群的适应度是 .1/.4/.7/.9；报出来的必须是 9.0 那个，不是被复制的 1.0。"""

        text = _evolve_and_capture(monkeypatch, generations=1)

        assert "- Intensity: 9.00" in text
        assert "- Intensity: 1.00" not in text


class TestAttackDefenseGameReportsItsOwnAttack:
    def test_play_round_returns_the_attack_params_it_used(self):
        game = AttackDefenseGame(num_clients=6, model_shape=(4, 8))

        result = game.play_round()

        assert result["attack_params"] == game.attack_params_history[-1]
        assert 1.5 <= result["attack_params"]["intensity"] <= 10.0

    def test_run_game_never_prints_a_zero_intensity(self):
        """强度取值范围是 [1.5, 10.0]，打印出 0.00 只可能是参数根本没传下去。"""

        game = AttackDefenseGame(num_clients=6, model_shape=(4, 8))

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            game.run_game(max_rounds=2)

        assert "intensity=0.00" not in stdout.getvalue()
        assert "intensity=" in stdout.getvalue()
