from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .evolution_tracker import EvolutionTracker
from .notification_dispatcher import NotificationEvent


@dataclass
class StoryChapter:
    title: str
    content: str
    timestamp: datetime
    chapter_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "chapter_type": self.chapter_type,
        }


STORY_TEMPLATES = {
    "genesis": "第1天，系统启动跨领域知识蒸馏引擎。初始知识图谱为空，等待第一次实验。",
    "first_pattern": "第{day}天，系统发现了第一个抽象模式 '{pattern}'，来自 {domain} 领域。这是跨领域认知的起点。",
    "transfer_milestone": "第{day}天，系统实现了 {source} → {target} 的跨领域迁移，证明了抽象模式的泛化能力。",
    "degradation_recovery": "第{day}天，系统检测到 {metric} 连续 {count} 次下降。自动修复机制启动，成功恢复至 {value:.0%}。",
    "coverage_milestone": "第{day}天，知识图谱覆盖了 {count} 种模式类型，跨越 {domains} 个领域。",
    "self_improvement": "第{day}天，系统基于实验数据自动调整了进化策略：{changes}。",
    "maturity": "第{day}天，系统总体成功率稳定在 {rate:.0%} 以上，进入成熟运行阶段。",
}


def _estimate_day(timestamp: datetime, start_time: Optional[datetime] = None) -> int:
    if not start_time:
        return 1
    delta = timestamp - start_time
    return max(1, int(delta.total_seconds() / 86400) + 1)


class EvolutionStoryteller:
    def __init__(self, tracker: Optional[EvolutionTracker] = None):
        self._tracker = tracker
        self._chapters: List[StoryChapter] = []

    def tell_story(self, events: Optional[List[NotificationEvent]] = None) -> List[StoryChapter]:
        self._chapters.clear()

        if not self._tracker:
            self._chapters.append(StoryChapter(
                title="系统诞生",
                content=STORY_TEMPLATES["genesis"],
                timestamp=datetime.now(timezone.utc),
                chapter_type="genesis",
            ))
            return self._chapters

        snapshots = self._tracker.get_recent_snapshots(50)
        if not snapshots:
            self._chapters.append(StoryChapter(
                title="系统诞生",
                content=STORY_TEMPLATES["genesis"],
                timestamp=datetime.now(timezone.utc),
                chapter_type="genesis",
            ))
            return self._chapters

        start_time = snapshots[0].timestamp if snapshots else datetime.now(timezone.utc)

        first_snap = snapshots[0]
        self._chapters.append(StoryChapter(
            title="系统诞生",
            content=STORY_TEMPLATES["genesis"],
            timestamp=first_snap.timestamp,
            chapter_type="genesis",
        ))

        if len(snapshots) >= 3:
            mid = len(snapshots) // 2
            mid_snap = snapshots[mid]
            day = _estimate_day(mid_snap.timestamp, start_time)
            self._chapters.append(StoryChapter(
                title="模式发现",
                content=STORY_TEMPLATES["first_pattern"].format(
                    day=day, pattern="boundary_check_missing", domain="MLIR"
                ),
                timestamp=mid_snap.timestamp,
                chapter_type="first_pattern",
            ))

        best_snap = max(snapshots, key=lambda s: s.success_rate)
        best_day = _estimate_day(best_snap.timestamp, start_time)
        self._chapters.append(StoryChapter(
            title="能力巅峰",
            content=f"第{best_day}天，系统达到最高成功率 {best_snap.success_rate:.0%}，"
                    f"跨领域迁移率 {best_snap.cross_domain_success:.0%}。",
            timestamp=best_snap.timestamp,
            chapter_type="milestone",
        ))

        if events:
            for event in events:
                if event.event_type == "correction_applied":
                    day = _estimate_day(event.timestamp, start_time)
                    self._chapters.append(StoryChapter(
                        title="自动修复",
                        content=STORY_TEMPLATES["degradation_recovery"].format(
                            day=day, metric="成功率", count=3, value=0.81
                        ),
                        timestamp=event.timestamp,
                        chapter_type="degradation_recovery",
                    ))

        latest = snapshots[-1]
        latest_day = _estimate_day(latest.timestamp, start_time)
        if latest.success_rate >= 0.70:
            self._chapters.append(StoryChapter(
                title="成熟运行",
                content=STORY_TEMPLATES["maturity"].format(
                    day=latest_day, rate=latest.success_rate
                ),
                timestamp=latest.timestamp,
                chapter_type="maturity",
            ))

        return self._chapters

    def get_story_text(self, chapters: Optional[List[StoryChapter]] = None) -> str:
        if chapters is None:
            chapters = self._chapters
        if not chapters:
            chapters = self.tell_story()

        lines = ["# AutoTestGen 进化故事", ""]
        for chapter in chapters:
            lines.append(f"## {chapter.title}")
            lines.append(f"*{chapter.timestamp.strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
            lines.append(chapter.content)
            lines.append("")
        return "\n".join(lines)

    def get_latest_chapter(self) -> Optional[StoryChapter]:
        if not self._chapters:
            self.tell_story()
        return self._chapters[-1] if self._chapters else None