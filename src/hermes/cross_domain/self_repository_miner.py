"""自身仓库演化阶段挖掘。

历史实现里有两条不可信的路径，现已被移除：

1. git 不可用时返回硬编码的“三个阶段”，连提交数（12/15/10）和
   commit_range（`v0.1-v0.5` 之类）都是凭空写死的；
2. git 可用时把提交历史**等分成三份**再套上硬编码阶段名，阶段名与实际提交
   内容无关（例如把 `ci_build` 占主导的阶段命名为“跨领域扩展期”）。

两种情况产出的“演化阶段”都会被下游当成真实实验数据引用，属于数据伪造。
现在本模块只做一件事：调用 `GitPhaseDetector` 从真实 git 历史推导阶段。
git 不可用时返回空列表，并通过 `git_unavailable` / `unavailable_reason`
显式告知调用方，绝不返回编造数据。
"""

from typing import Any, Dict, List, Optional

from .git_phase_detector import EvolutionPhase, GitPhaseDetector, PhaseDetectionResult

#: 历史名称，保留以兼容既有调用方；现在与 `EvolutionPhase` 是同一个类，
#: 阶段划分算法只有一套（`GitPhaseDetector`）。
EvolutionStage = EvolutionPhase


class SelfRepositoryMiner:
    """用真实 git 历史划分自身演化阶段（`GitPhaseDetector` 的适配器）。"""

    def __init__(self, repo_path: str = ".", detector: Optional[GitPhaseDetector] = None):
        self._repo_path = repo_path
        self._detector = detector or GitPhaseDetector(repo_path)
        self._stages: List[EvolutionStage] = []
        self._result: Optional[PhaseDetectionResult] = None

    def mine_self(self) -> List[EvolutionStage]:
        """返回由真实 git 历史推导的阶段；git 不可用时返回空列表。"""

        self._result = self._detect()
        self._stages = list(self._result.phases)
        return self._stages

    def get_stages(self) -> List[EvolutionStage]:
        return self._stages

    @property
    def git_unavailable(self) -> bool:
        """git 历史是否不可用；为 True 时阶段列表必为空。"""

        return bool(self._result and self._result.git_unavailable)

    @property
    def unavailable_reason(self) -> str:
        return self._result.reason if self._result else ""

    @property
    def detection(self) -> Optional[PhaseDetectionResult]:
        return self._result

    def to_dict(self) -> Dict[str, Any]:
        if self._result is None:
            # 尚未挖掘过：先做一次检测，避免把“还没查过”报成“git 可用”
            self.mine_self()
        result = self._result
        return {
            "repo_path": self._repo_path,
            "git_unavailable": result.git_unavailable,
            "reason": result.reason,
            "method": result.method,
            "commit_count": result.commit_count,
            "stages": [stage.to_dict() for stage in self._stages],
        }

    def _detect(self) -> PhaseDetectionResult:
        try:
            return self._detector.detect()
        except Exception as error:  # pragma: no cover - 防御性分支
            return PhaseDetectionResult(
                git_unavailable=True,
                reason=f"阶段检测失败：{error}",
                method="unavailable",
            )
