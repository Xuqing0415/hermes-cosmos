"""对抗性联邦学习的**成功判据**测试。

这一套测的不是防御好不好用，而是「凭什么敢说自己成功」。原来的判据只有
``acc_drop < 10`` 一条：模型停在随机水平（10 类 = 0.10）时，准确率本来就不会掉，
于是「一次攻击都没发生、模型什么都没学到」的运行会被印成
``SUCCESS: Defense effectively mitigated attacks!``。

所以这里断言的是**证据门槛**：基线得学过东西、得真的发生过攻击、防御后的准确率
不能低于基线也不能仍在随机水平 —— 四条都过才允许出现 SUCCESS 字样。
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes_unified.federated.run_adversarial_fl import (  # noqa: E402  (须先补 sys.path)
    classify_defense_result,
    describe_outcome,
)

NUM_CLASSES = 10
CHANCE = 0.1


class TestClassifyDefenseResult:
    def test_chance_level_run_without_attacks_is_inconclusive(self):
        """真实跑出来的那一次：acc=0.09、攻击 0 次 —— 以前被印成 SUCCESS。"""

        verdict, reasons = classify_defense_result(0.09, 0.09, 0, NUM_CLASSES)

        assert verdict == "INCONCLUSIVE"
        assert any("baseline_acc" in reason for reason in reasons)
        assert any("attack_events=0" in reason for reason in reasons)

    def test_a_defense_that_holds_up_is_success(self):
        verdict, reasons = classify_defense_result(0.80, 0.85, 5, NUM_CLASSES)

        assert verdict == "SUCCESS"
        assert reasons == []

    def test_defense_worse_than_baseline_is_not_success(self):
        verdict, reasons = classify_defense_result(0.80, 0.70, 5, NUM_CLASSES)

        assert verdict == "INCONCLUSIVE"
        assert any("defended_acc" in reason and "baseline_acc" in reason for reason in reasons)

    def test_learned_baseline_without_any_attack_is_inconclusive(self):
        """基线学得不错，但这一轮压根没有攻击 —— 没有可防御的对象，不算证明了防御。"""

        verdict, reasons = classify_defense_result(0.85, 0.85, 0, NUM_CLASSES)

        assert verdict == "INCONCLUSIVE"
        assert any("attack_events=0" in reason for reason in reasons)

    def test_defended_model_still_at_chance_is_inconclusive(self):
        verdict, reasons = classify_defense_result(0.80, 0.11, 5, NUM_CLASSES)

        assert verdict == "INCONCLUSIVE"
        assert any("still at chance level" in reason for reason in reasons)

    def test_chance_threshold_scales_with_the_number_of_classes(self):
        """门槛是 1.5 倍随机水平，所以同一个准确率在不同类别数下结论不同。

        0.70 在 10 类问题上远高于门槛（0.15），在 2 类问题上却低于门槛（0.75）。
        """

        assert classify_defense_result(0.70, 0.72, 3, 10)[0] == "SUCCESS"
        assert classify_defense_result(0.70, 0.72, 3, 2)[0] == "INCONCLUSIVE"


class TestDescribeOutcome:
    def test_inconclusive_run_never_prints_success(self):
        text = "\n".join(describe_outcome(0.09, 0.09, 0, NUM_CLASSES))

        assert "INCONCLUSIVE" in text
        assert "SUCCESS" not in text

    def test_mitigation_claim_never_co_occurs_with_chance_level_accuracy(self):
        """被修掉的那种自相矛盾：随机水平的准确率 + 「有效缓解了攻击」。"""

        for baseline, defended, events in [
            (0.09, 0.09, 0),
            (0.10, 0.10, 2),
            (0.12, 0.08, 4),
            (0.11, 0.11, 0),
        ]:
            text = "\n".join(describe_outcome(baseline, defended, events, NUM_CLASSES))
            assert "effectively mitigated" not in text
            assert "SUCCESS" not in text
            assert "INCONCLUSIVE" in text

    def test_success_line_is_the_only_place_that_claims_mitigation(self):
        text = "\n".join(describe_outcome(0.80, 0.85, 5, NUM_CLASSES))

        assert "SUCCESS" in text
        assert "effectively mitigated" in text

    def test_reasons_are_reported_per_line(self):
        lines = describe_outcome(0.09, 0.09, 0, NUM_CLASSES)

        assert len(lines) >= 3
        assert all(line.startswith("   - ") for line in lines[1:])
