# 本目录是模拟（simulation），不是真实联邦训练

`hermes_unified/self_evolution/` 里的代码跑的是**模拟实验**。

- `SelfEvolvingFederatedLearning.run_single_experiment()` 用合成任务与带噪声的预测值
  代替真实 FL 训练（源码里的注释就是 `# Simulate experiment` / `# For demo purposes`）。
- 因此 `self_evolution_results/evolution_progress.json`、`best_deployment.json` 里的
  精度、轮次等数字只是演示数据，**不能**作为「系统具备真实联邦学习能力」的证据。
- `run_self_evolution.py` 的 docstring 与 CLI 输出的第一行也会自报这一点。

## 机器可读的标记

```python
from hermes_unified.self_evolution import IS_SIMULATION  # -> True
```

## 下游约定

`hermes.self_research` 采集到本模块产出的数据时，必须把它标成 `SYNTHETIC`
（不要标 `REAL`）。行级数据的自报方式见 `hermes.cross_domain.evolution_tracker`：

```python
from hermes.cross_domain import SIMULATED_SOURCE, SOURCE_METADATA_KEY
metadata = {SOURCE_METADATA_KEY: SIMULATED_SOURCE, ...}
```

`ResearchDataCollector` 会读取这个标记，把 `snapshots` / `pattern_success_rates` /
`mental_model` / `similarity_pairs` 从 `REAL` 降级为 `SYNTHETIC`，证据等级随之降到 C。
