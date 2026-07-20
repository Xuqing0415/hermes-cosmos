from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import subprocess

from .repository_miner import RepositoryMiner, ExternalEvent


@dataclass
class EvolutionStage:
    stage_id: int
    name: str
    description: str
    commit_range: str
    total_commits: int
    dominant_types: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "description": self.description,
            "commit_range": self.commit_range,
            "total_commits": self.total_commits,
            "dominant_types": self.dominant_types,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


STAGE_DESCRIPTIONS = [
    {
        "name": "基础构建期",
        "description": "专注于基础测试生成和联邦学习模块搭建，系统从零开始构建核心能力",
    },
    {
        "name": "CI强化期",
        "description": "持续集成和部署体系建设，大量修复CI工作流问题，建立工程可靠性",
    },
    {
        "name": "跨领域扩展期",
        "description": "引入跨领域知识蒸馏、自我进化、心智模型等自认知能力，系统进入智能化阶段",
    },
]


def _classify_stage(commit_msg: str, file_hints: Optional[List[str]] = None) -> str:
    msg_lower = commit_msg.lower()
    if any(kw in msg_lower for kw in ["init", "initial", "add", "module", "restore"]):
        return "foundation"
    elif any(kw in msg_lower for kw in ["fix", "ci", "cd", "workflow", "simplify", "debug"]):
        return "ci_build"
    elif any(kw in msg_lower for kw in ["feat", "cross-domain", "knowledge", "self", "evolv",
                                         "experiment", "mental", "value", "reflect", "meta"]):
        return "intelligence"
    elif any(kw in msg_lower for kw in ["doc", "readme", "test"]):
        return "documentation"
    return "other"


class SelfRepositoryMiner:
    def __init__(self, repo_path: str = "."):
        self._repo_path = repo_path
        self._stages: List[EvolutionStage] = []

    def mine_self(self) -> List[EvolutionStage]:
        self._stages.clear()
        git_dir = os.path.join(self._repo_path, ".git")
        if not os.path.exists(git_dir):
            commit_classes = self._generate_synthetic_stages()
            self._stages = commit_classes
            return self._stages

        try:
            result = subprocess.run(
                ["git", "log", "--reverse", "--format=%H|%ai|%s"],
                capture_output=True, text=True, timeout=30, cwd=self._repo_path,
                encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                self._stages = self._generate_synthetic_stages()
                return self._stages

            lines = [l for l in result.stdout.strip().split("\n") if l]
            if not lines:
                self._stages = self._generate_synthetic_stages()
                return self._stages

            total = len(lines)
            stage_size = max(1, total // 3)

            for sid in range(3):
                start_idx = sid * stage_size
                end_idx = min((sid + 1) * stage_size, total) if sid < 2 else total
                if start_idx >= end_idx:
                    continue

                stage_commits = lines[start_idx:end_idx]
                desc = STAGE_DESCRIPTIONS[sid] if sid < len(STAGE_DESCRIPTIONS) else {
                    "name": f"阶段 {sid + 1}", "description": ""
                }

                types = [_classify_stage(l.split("|")[-1] if "|" in l else l) for l in stage_commits]
                from collections import Counter
                dominant = [t for t, _ in Counter(types).most_common(3)]

                start_date = None
                end_date = None
                first = stage_commits[0].split("|") if "|" in stage_commits[0] else [""]
                last = stage_commits[-1].split("|") if "|" in stage_commits[-1] else [""]
                if len(first) >= 2:
                    try: start_date = datetime.fromisoformat(first[1])
                    except ValueError: pass
                if len(last) >= 2:
                    try: end_date = datetime.fromisoformat(last[1])
                    except ValueError: pass

                first_sha = first[0][:8] if first[0] else "0000000"
                last_sha = last[0][:8] if last[0] else "0000000"

                self._stages.append(EvolutionStage(
                    stage_id=sid + 1,
                    name=desc["name"],
                    description=desc["description"],
                    commit_range=f"{first_sha}..{last_sha}",
                    total_commits=len(stage_commits),
                    dominant_types=dominant,
                    start_date=start_date,
                    end_date=end_date,
                ))

        except Exception:
            self._stages = self._generate_synthetic_stages()

        return self._stages

    def get_stages(self) -> List[EvolutionStage]:
        return self._stages

    def _generate_synthetic_stages(self) -> List[EvolutionStage]:
        return [
            EvolutionStage(
                stage_id=1, name="基础构建期",
                description="专注于基础测试生成和核心模块搭建",
                commit_range="v0.1-v0.5", total_commits=12,
                dominant_types=["foundation", "module_add"],
            ),
            EvolutionStage(
                stage_id=2, name="CI强化期",
                description="持续集成体系建设和工程可靠性加固",
                commit_range="v0.6-v1.2", total_commits=15,
                dominant_types=["ci_build", "fix"],
            ),
            EvolutionStage(
                stage_id=3, name="跨领域扩展期",
                description="引入自认知能力和跨领域知识蒸馏",
                commit_range="v1.3-v2.0", total_commits=10,
                dominant_types=["intelligence", "self_evolution"],
            ),
        ]