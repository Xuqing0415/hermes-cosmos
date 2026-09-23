"""基于真实 git 历史划分演化阶段。

与 `SelfRepositoryMiner` 的旧实现相比，这里做了三件不同的事：

1. 阶段边界来自提交内容的真实变化（主导提交类型翻转、提交时间间隔），
   不再把历史简单地等分成三份；
2. 阶段名由该阶段的主导提交类型生成，不使用硬编码标签；
3. git 不可用时返回 `git_unavailable=True` 的空结果，**不再伪造阶段**。

整个算法是确定性的：同一段历史必然得到同一套阶段划分。
"""

import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

GIT_LOG_FORMAT = "%H|%ai|%s"
GIT_LOG_TIMEOUT = 30

CATEGORY_LABELS = {
    "foundation": "基础构建",
    "ci_build": "CI与修复",
    "intelligence": "智能扩展",
    "documentation": "文档",
    "other": "其他",
}

# 同分时的裁决顺序，保证结果确定
CATEGORY_PRIORITY = ("foundation", "ci_build", "intelligence", "documentation", "other")

CONVENTIONAL_RE = re.compile(r"^\s*([a-zA-Z]+)\s*(?:\(([^)]*)\))?!?\s*:\s*(.*)$")

CONVENTIONAL_TYPES = {
    "fix": "ci_build",
    "ci": "ci_build",
    "build": "ci_build",
    "test": "ci_build",
    "docs": "documentation",
    "doc": "documentation",
    "chore": "other",
    "refactor": "other",
    "style": "other",
    "perf": "other",
    "revert": "other",
}

KEYWORDS = {
    "foundation": (
        "init",
        "initial",
        "bootstrap",
        "scaffold",
        "setup",
        "skeleton",
        "module",
        "modules",
        "implement",
        "introduce",
        "structure",
        "scheduler",
        "gateway",
        "checkpoint",
        "初始化",
        "搭建",
        "模块",
        "引入",
        "基础",
    ),
    "ci_build": (
        "fix",
        "ci",
        "cd",
        "workflow",
        "lint",
        "flake8",
        "black",
        "isort",
        "build",
        "release",
        "docker",
        "deploy",
        "depend",
        "pytest",
        "test",
        "repair",
        "bug",
        "workaround",
        "修复",
        "门禁",
        "流水线",
        "构建",
        "依赖",
        "测试",
        "部署",
        "回归",
    ),
    "intelligence": (
        "cross",
        "domain",
        "knowledge",
        "self",
        "evolv",
        "mental",
        "value",
        "reflect",
        "meta",
        "introspect",
        "cognit",
        "memory",
        "reason",
        "research",
        "paper",
        "empiric",
        "distill",
        "跨领域",
        "知识",
        "自我",
        "进化",
        "心智",
        "反思",
        "认知",
        "元认知",
        "自研究",
        "论文",
        "蒸馏",
    ),
    "documentation": (
        "doc",
        "docs",
        "readme",
        "comment",
        "annotate",
        "changelog",
        "文档",
        "注释",
        "说明",
    ),
}

# 允许的英文屈折后缀，避免 doc 匹配到 docker、test 匹配到 latest
INFLECTIONS = ("s", "es", "ed", "ing", "d")

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _token_matches(token: str, keyword: str) -> bool:
    """token 是否命中关键词（英文支持常见屈折，中文按子串）。"""

    if CJK_RE.search(keyword):
        return keyword in token
    if token == keyword:
        return True
    if token.startswith(keyword) and token[len(keyword) :] in INFLECTIONS:
        return True
    return False


def classify_commit(subject: str) -> str:
    """把一条提交主题映射到主导类型。

    先看 conventional commit 前缀（本仓库大量使用），再用关键词打分兜底；
    同分时按 CATEGORY_PRIORITY 顺序裁决。
    """

    text = (subject or "").strip()
    commit_type = ""
    scope = ""
    match = CONVENTIONAL_RE.match(text)
    if match:
        commit_type = match.group(1).lower()
        scope = (match.group(2) or "").strip().lower()

    scores: Dict[str, float] = dict.fromkeys(CATEGORY_PRIORITY, 0.0)
    tokens = TOKEN_RE.findall(text.lower())

    for index, token in enumerate(tokens):
        weight = 2.0 if index < 3 else 1.0
        for category, keywords in KEYWORDS.items():
            if any(_token_matches(token, keyword) for keyword in keywords):
                scores[category] += weight

    if scope:
        scoped = CONVENTIONAL_TYPES.get(scope)
        if scoped:
            scores[scoped] += 2.0

    if commit_type in CONVENTIONAL_TYPES:
        scores[CONVENTIONAL_TYPES[commit_type]] += 1.5
    if commit_type == "feat" and scores["intelligence"] > 0:
        scores["intelligence"] += 2.0

    if not any(scores.values()):
        return "other"
    return max(scores.items(), key=lambda item: (item[1], -CATEGORY_PRIORITY.index(item[0])))[0]


@dataclass
class CommitRecord:
    sha: str
    date: Optional[datetime]
    subject: str
    category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha": self.sha,
            "date": self.date.isoformat() if self.date else None,
            "subject": self.subject,
            "category": self.category,
        }


@dataclass
class EvolutionPhase:
    stage_id: int
    name: str
    description: str
    commit_range: str
    total_commits: int
    dominant_types: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    category: str = "other"
    category_share: float = 0.0
    boundary_signal: str = "theme_change"

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
            "category": self.category,
            "category_share": round(self.category_share, 4),
            "boundary_signal": self.boundary_signal,
        }


@dataclass
class PhaseDetectionResult:
    phases: List[EvolutionPhase] = field(default_factory=list)
    git_unavailable: bool = False
    reason: str = ""
    commit_count: int = 0
    method: str = "unavailable"
    boundaries: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": [phase.to_dict() for phase in self.phases],
            "git_unavailable": self.git_unavailable,
            "reason": self.reason,
            "commit_count": self.commit_count,
            "method": self.method,
            "boundaries": self.boundaries,
        }


def parse_git_log(stdout: str) -> List[CommitRecord]:
    """解析 `git log --format=%H|%ai|%s` 的输出（按提交时间从早到晚）。"""

    commits: List[CommitRecord] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha, raw_date, subject = parts[0], parts[1], parts[2]
        commits.append(
            CommitRecord(
                sha=sha,
                date=parse_git_date(raw_date),
                subject=subject,
                category=classify_commit(subject),
            )
        )
    return commits


def parse_git_date(raw_date: str) -> Optional[datetime]:
    """解析 git `%ai` / `%ci` 时间戳（形如 `2026-01-02 10:20:30 +0800`）。

    不能直接用 `datetime.fromisoformat`：Python 3.10 及更早版本不接受
    `+0800` 这种紧凑时区写法（3.11 才放宽），那会让所有提交时间静默变成 None，
    进而让阶段边界丢失时间信号。这里先用 `%z` 显式解析。
    """

    text = (raw_date or "").strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class GitPhaseDetector:
    """用真实 git 历史推导演化阶段；不做任何伪造。"""

    def __init__(
        self,
        repo_path: str = ".",
        max_phases: int = 6,
        since: Optional[str] = None,
        min_phase_ratio: float = 0.08,
    ):
        self._repo_path = repo_path
        self._max_phases = max(1, max_phases)
        self._since = since
        self._min_phase_ratio = min_phase_ratio

    # ------------------------------------------------------------------ 入口

    def detect(self) -> PhaseDetectionResult:
        git_dir = os.path.join(self._repo_path, ".git")
        if not os.path.isdir(git_dir):
            return PhaseDetectionResult(
                git_unavailable=True,
                reason=f"未找到 git 仓库（{self._repo_path}）",
                method="unavailable",
            )

        command = ["git", "log", "--reverse", f"--format={GIT_LOG_FORMAT}"]
        if self._since:
            command.append(f"--since={self._since}")

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=GIT_LOG_TIMEOUT,
                cwd=self._repo_path,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as error:
            return PhaseDetectionResult(
                git_unavailable=True,
                reason=f"git 调用失败：{error}",
                method="unavailable",
            )

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip().splitlines()
            return PhaseDetectionResult(
                git_unavailable=True,
                reason=f"git log 返回 {completed.returncode}：{detail[0] if detail else 'unknown'}",
                method="unavailable",
            )

        commits = parse_git_log(completed.stdout)
        if not commits:
            return PhaseDetectionResult(
                git_unavailable=True,
                reason="git 历史为空",
                method="unavailable",
            )

        return self.detect_from_commits(commits)

    # ------------------------------------------------------------------ 算法

    def detect_from_commits(self, commits: Sequence[CommitRecord]) -> PhaseDetectionResult:
        total = len(commits)
        if total == 0:
            return PhaseDetectionResult(git_unavailable=True, reason="无提交记录", method="unavailable")

        min_stage = max(2, int(round(total * self._min_phase_ratio)))
        if total < min_stage * 2:
            phases = self._build_phases(commits, [0], ["single"])
            return PhaseDetectionResult(
                phases=phases,
                commit_count=total,
                method="single_phase",
                boundaries=[],
            )

        window = max(1, total // 15)
        smoothed = self._smooth([commit.category for commit in commits], window)

        boundaries = self._theme_boundaries(smoothed, min_stage)
        boundaries = self._merge_boundaries(boundaries, self._gap_boundaries(commits, min_stage), min_stage)
        starts, signals = self._assemble_starts(boundaries, commits)
        starts, signals = self._merge_same_category(starts, signals, commits)
        starts, signals = self._enforce_max_phases(starts, signals, commits)

        phases = self._build_phases(commits, starts, signals)
        return PhaseDetectionResult(
            phases=phases,
            commit_count=total,
            method="git_log_theme_and_gap" if len(phases) > 1 else "single_phase",
            boundaries=starts[1:],
        )

    @staticmethod
    def _smooth(categories: Sequence[str], window: int) -> List[str]:
        smoothed: List[str] = []
        for index in range(len(categories)):
            chunk = categories[max(0, index - window) : index + window + 1]
            counter = Counter(chunk)
            smoothed.append(max(counter.items(), key=lambda item: (item[1], -CATEGORY_PRIORITY.index(item[0])))[0])
        return smoothed

    @staticmethod
    def _theme_boundaries(smoothed: Sequence[str], min_stage: int) -> List[int]:
        total = len(smoothed)
        boundaries: List[int] = []
        last = 0
        index = min_stage

        while index < total:
            if smoothed[index] != smoothed[index - 1]:
                run = 1
                cursor = index + 1
                while cursor < total and smoothed[cursor] == smoothed[index]:
                    run += 1
                    cursor += 1
                if run >= min_stage and index - last >= min_stage and total - index >= min_stage:
                    boundaries.append(index)
                    last = index
                    index = cursor
                    continue
            index += 1

        return boundaries

    @staticmethod
    def _gap_boundaries(commits: Sequence[CommitRecord], min_stage: int) -> List[int]:
        dated = [(index, commit.date) for index, commit in enumerate(commits) if commit.date is not None]
        if len(dated) < 4:
            return []

        gaps = [(dated[i + 1][1] - dated[i][1]).total_seconds() / 86400.0 for i in range(len(dated) - 1)]
        positive = sorted(gap for gap in gaps if gap > 0)
        if not positive:
            return []

        median_gap = positive[len(positive) // 2]
        threshold = max(7.0, median_gap * 4.0)

        boundaries: List[int] = []
        last = 0
        for i, gap in enumerate(gaps):
            position = dated[i + 1][0]
            if gap > threshold and position - last >= min_stage and len(commits) - position >= min_stage:
                boundaries.append(position)
                last = position
        return boundaries

    @staticmethod
    def _merge_boundaries(primary: Sequence[int], extra: Sequence[int], min_stage: int) -> List[int]:
        merged: List[int] = []
        for candidate in sorted(set(primary) | set(extra)):
            if not merged or candidate - merged[-1] >= min_stage:
                merged.append(candidate)
        return merged

    @staticmethod
    def _assemble_starts(boundaries: Sequence[int], commits: Sequence[CommitRecord]) -> Tuple[List[int], List[str]]:
        starts = [0]
        signals = ["start"]
        for boundary in boundaries:
            starts.append(boundary)
            signals.append("theme_change")
        return starts, signals

    @staticmethod
    def _dominant_category(commits: Sequence[CommitRecord], start: int, end: int) -> str:
        counter = Counter(commit.category for commit in commits[start:end])
        if not counter:
            return "other"
        return max(counter.items(), key=lambda item: (item[1], -CATEGORY_PRIORITY.index(item[0])))[0]

    def _merge_same_category(
        self,
        starts: List[int],
        signals: List[str],
        commits: Sequence[CommitRecord],
    ) -> Tuple[List[int], List[str]]:
        """相邻阶段主导类型相同则合并——类型没变的切分不是真正的阶段边界。"""

        total = len(commits)
        starts = list(starts)
        signals = list(signals)

        merged = True
        while merged and len(starts) > 1:
            merged = False
            for index in range(1, len(starts)):
                end = starts[index + 1] if index + 1 < len(starts) else total
                left = self._dominant_category(commits, starts[index - 1], starts[index])
                right = self._dominant_category(commits, starts[index], end)
                if left == right:
                    del starts[index]
                    del signals[index]
                    merged = True
                    break

        return starts, signals

    def _enforce_max_phases(
        self,
        starts: List[int],
        signals: List[str],
        commits: Sequence[CommitRecord],
    ) -> Tuple[List[int], List[str]]:
        total = len(commits)
        while len(starts) > self._max_phases:
            sizes = [
                (starts[index + 1] if index + 1 < len(starts) else total) - starts[index]
                for index in range(len(starts))
            ]
            smallest = min(range(len(sizes)), key=lambda index: (sizes[index], index))
            drop = smallest if smallest > 0 else 1
            del starts[drop]
            del signals[drop]
        return starts, signals

    @staticmethod
    def _build_phases(
        commits: Sequence[CommitRecord],
        starts: Sequence[int],
        signals: Sequence[str],
    ) -> List[EvolutionPhase]:
        total = len(commits)
        phases: List[EvolutionPhase] = []

        for order, start in enumerate(starts):
            end = starts[order + 1] if order + 1 < len(starts) else total
            chunk = commits[start:end]
            if not chunk:
                continue

            counter = Counter(commit.category for commit in chunk)
            dominant, count = max(
                counter.items(),
                key=lambda item: (item[1], -CATEGORY_PRIORITY.index(item[0])),
            )
            share = count / len(chunk)
            dominant_types = [
                category
                for category, _ in sorted(
                    counter.items(),
                    key=lambda item: (-item[1], CATEGORY_PRIORITY.index(item[0])),
                )[:3]
            ]

            start_date = chunk[0].date
            end_date = chunk[-1].date
            label = CATEGORY_LABELS.get(dominant, dominant)
            span = "未知时间"
            if start_date is not None and end_date is not None:
                span = f"{start_date.date().isoformat()} → {end_date.date().isoformat()}"

            phases.append(
                EvolutionPhase(
                    stage_id=len(phases) + 1,
                    name=f"{label}期",
                    description=(f"{len(chunk)} 次提交；主导类型 {dominant}({share:.0%})；时间 {span}"),
                    commit_range=f"{chunk[0].sha[:8]}..{chunk[-1].sha[:8]}",
                    total_commits=len(chunk),
                    dominant_types=dominant_types,
                    start_date=start_date,
                    end_date=end_date,
                    category=dominant,
                    category_share=share,
                    boundary_signal=signals[order] if order < len(signals) else "theme_change",
                )
            )

        return phases
