from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import re
import random


@dataclass
class ExternalEvent:
    event_id: str
    repo_name: str
    commit_sha: Optional[str]
    timestamp: datetime
    event_type: str
    description: str
    files_changed: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    pattern_type: Optional[str] = None
    resolution_days: Optional[float] = None
    severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "repo_name": self.repo_name,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "files_changed": self.files_changed[:5],
            "domain": self.domain,
            "pattern_type": self.pattern_type,
            "resolution_days": self.resolution_days,
            "severity": self.severity,
        }


@dataclass
class RepoMiningReport:
    repo_url: str
    repo_name: str
    total_events: int
    events_by_type: Dict[str, int]
    events_by_domain: Dict[str, int]
    time_span_days: int
    events: List[ExternalEvent] = field(default_factory=list)
    mining_method: str = "simulated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "repo_name": self.repo_name,
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_domain": self.events_by_domain,
            "time_span_days": self.time_span_days,
            "mining_method": self.mining_method,
        }


# Synthetic demo events that simulate what would be mined from a real repo
DEMO_EVENTS_DATA = [
    {
        "event_type": "boundary_fix",
        "domains": ["python"],
        "description": "修复 DataFrame.iloc 越界访问，添加索引范围检查",
        "pattern_type": "boundary_check_missing",
        "severity": "high",
    },
    {
        "event_type": "boundary_fix",
        "domains": ["python"],
        "description": "修复 Series.__getitem__ 负索引越界问题",
        "pattern_type": "boundary_check_missing",
        "severity": "high",
    },
    {
        "event_type": "type_check_improvement",
        "domains": ["python"],
        "description": "增强 merge/join 操作的类型推断",
        "pattern_type": "type_mismatch",
        "severity": "medium",
    },
    {
        "event_type": "type_check_improvement",
        "domains": ["python"],
        "description": "添加 read_csv dtype 参数校验",
        "pattern_type": "input_validation_missing",
        "severity": "medium",
    },
    {
        "event_type": "performance_opt",
        "domains": ["python"],
        "description": "优化 DataFrame.groupby 聚合性能 40%",
        "pattern_type": "performance_bottleneck",
        "severity": "low",
    },
    {
        "event_type": "performance_opt",
        "domains": ["python"],
        "description": "重写 concat 操作的底层内存分配策略",
        "pattern_type": "performance_bottleneck",
        "severity": "medium",
    },
    {
        "event_type": "api_deprecation",
        "domains": ["python"],
        "description": "弃用 DataFrame.append，推荐使用 DataFrame.concat",
        "pattern_type": "api_version_deprecated",
        "severity": "medium",
    },
    {
        "event_type": "api_deprecation",
        "domains": ["python"],
        "description": "移除旧的 Panel 数据结构",
        "pattern_type": "api_version_deprecated",
        "severity": "high",
    },
    {
        "event_type": "memory_opt",
        "domains": ["python"],
        "description": "优化 DataFrame 内存使用，引入 BlockManager 压缩",
        "pattern_type": "memory_leak",
        "severity": "medium",
    },
    {
        "event_type": "memory_opt",
        "domains": ["python"],
        "description": "添加内存使用监控，防止大数据集 OOM",
        "pattern_type": "resource_limit_missing",
        "severity": "high",
    },
    {
        "event_type": "recursion_fix",
        "domains": ["python"],
        "description": "修复 to_dict 递归深度过高导致栈溢出",
        "pattern_type": "infinite_recursion",
        "severity": "high",
    },
    {
        "event_type": "concurrency_fix",
        "domains": ["python"],
        "description": "修复并行读取时的竞态条件",
        "pattern_type": "concurrency_race_condition",
        "severity": "high",
    },
    {
        "event_type": "boundary_fix",
        "domains": ["mlir", "compiler"],
        "description": "修复 memref.load 越界检查遗漏的边界情况",
        "pattern_type": "boundary_check_missing",
        "severity": "high",
    },
    {
        "event_type": "validation_add",
        "domains": ["mlir", "compiler"],
        "description": "添加操作数类型约束验证 pass",
        "pattern_type": "input_validation_missing",
        "severity": "medium",
    },
    {
        "event_type": "validation_add",
        "domains": ["kubernetes"],
        "description": "添加 ValidatingAdmissionPolicy 资源校验",
        "pattern_type": "input_validation_missing",
        "severity": "medium",
    },
]


class RepositoryMiner:
    def __init__(self):
        self._events: List[ExternalEvent] = []

    def mine(self, repo_url: str, max_events: int = 50) -> RepoMiningReport:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        if repo_url == "DEMO":
            return self._generate_demo_report()

        # Try local git log mining
        local_report = self._try_local_mining(repo_url)
        if local_report:
            return local_report

        # Fallback to demo report
        return self._generate_demo_report(repo_name)

    def _try_local_mining(self, repo_path: str) -> Optional[RepoMiningReport]:
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return None
        try:
            import subprocess
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=3.years", "--format=%H|%ai|%s"],
                capture_output=True, text=True, timeout=30, cwd=repo_path
            )
            if result.returncode != 0:
                return None

            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if not lines:
                return None

            events = []
            for line in lines[:max_events]:
                parts = line.split("|", 2)
                if len(parts) < 3:
                    continue
                sha, date_str, msg = parts
                try:
                    ts = datetime.fromisoformat(date_str)
                except ValueError:
                    ts = datetime.now(timezone.utc)

                event_type = self._classify_commit(msg)
                domain = self._detect_domain_from_msg(msg)

                events.append(ExternalEvent(
                    event_id=f"ext-{sha[:8]}" if sha else f"ext-{len(events)}",
                    repo_name=os.path.basename(repo_path),
                    commit_sha=sha[:8] if sha else None,
                    timestamp=ts,
                    event_type=event_type,
                    description=msg[:120],
                    domain=domain,
                    pattern_type=self._map_to_pattern(event_type),
                ))

            if events:
                type_counts = {}
                domain_counts = {}
                for e in events:
                    type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
                    d = e.domain or "unknown"
                    domain_counts[d] = domain_counts.get(d, 0) + 1

                times = [e.timestamp for e in events]
                time_span = (max(times) - min(times)).days if len(times) >= 2 else 365

                self._events = events
                return RepoMiningReport(
                    repo_url=repo_path,
                    repo_name=os.path.basename(repo_path),
                    total_events=len(events),
                    events_by_type=type_counts,
                    events_by_domain=domain_counts,
                    time_span_days=time_span,
                    events=events,
                    mining_method="git_log",
                )
        except Exception:
            pass
        return None

    def _generate_demo_report(self, repo_name: str = "external_repo") -> RepoMiningReport:
        rng = random.Random(42)
        now = datetime.now(timezone.utc)
        events = []

        for i, data in enumerate(DEMO_EVENTS_DATA):
            days_ago = rng.randint(0, 1095)
            ts = datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=timezone.utc)
            ts = ts.replace(day=max(1, min(28, ts.day - (days_ago // 30))))
            ts = ts.replace(hour=rng.randint(0, 23), minute=rng.randint(0, 59))

            resolution = round(rng.uniform(1, 30), 1) if data["severity"] != "low" else None

            events.append(ExternalEvent(
                event_id=f"ext-demo-{i + 1:04d}",
                repo_name=repo_name,
                commit_sha=f"{rng.randint(0, 0xFFFFFFFF):08x}",
                timestamp=ts,
                event_type=data["event_type"],
                description=data["description"],
                domain=data["domains"][0],
                pattern_type=data["pattern_type"],
                resolution_days=resolution,
                severity=data["severity"],
            ))

        type_counts = {}
        domain_counts = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            d = e.domain or "unknown"
            domain_counts[d] = domain_counts.get(d, 0) + 1

        self._events = events
        return RepoMiningReport(
            repo_url=repo_name,
            repo_name=repo_name,
            total_events=len(events),
            events_by_type=type_counts,
            events_by_domain=domain_counts,
            time_span_days=1095,
            events=events,
            mining_method="simulated",
        )

    def get_events(self) -> List[ExternalEvent]:
        return self._events

    def get_events_by_type(self, event_type: str) -> List[ExternalEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_domain(self, domain: str) -> List[ExternalEvent]:
        return [e for e in self._events if e.domain == domain]

    def _classify_commit(self, msg: str) -> str:
        msg_lower = msg.lower()
        if any(kw in msg_lower for kw in ["bound", "overflow", "index", "out of range"]):
            return "boundary_fix"
        elif any(kw in msg_lower for kw in ["type", "dtype", "cast", "infer"]):
            return "type_check_improvement"
        elif any(kw in msg_lower for kw in ["performance", "optimize", "speed", "fast"]):
            return "performance_opt"
        elif any(kw in msg_lower for kw in ["deprecat", "old", "remove"]):
            return "api_deprecation"
        elif any(kw in msg_lower for kw in ["memory", "leak", "oom"]):
            return "memory_opt"
        elif any(kw in msg_lower for kw in ["recursion", "stack", "deep"]):
            return "recursion_fix"
        elif any(kw in msg_lower for kw in ["concurrency", "race", "thread", "parallel"]):
            return "concurrency_fix"
        elif any(kw in msg_lower for kw in ["validat", "check", "assert", "verify"]):
            return "validation_add"
        return "other"

    def _detect_domain_from_msg(self, msg: str) -> str:
        msg_lower = msg.lower()
        if any(kw in msg_lower for kw in ["k8s", "kubernetes", "pod", "deployment"]):
            return "kubernetes"
        elif any(kw in msg_lower for kw in ["mlir", "dialect", "pass", "memref"]):
            return "mlir"
        return "python"

    def _map_to_pattern(self, event_type: str) -> Optional[str]:
        mapping = {
            "boundary_fix": "boundary_check_missing",
            "type_check_improvement": "type_mismatch",
            "performance_opt": "performance_bottleneck",
            "api_deprecation": "api_version_deprecated",
            "memory_opt": "memory_leak",
            "recursion_fix": "infinite_recursion",
            "concurrency_fix": "concurrency_race_condition",
            "validation_add": "input_validation_missing",
        }
        return mapping.get(event_type)