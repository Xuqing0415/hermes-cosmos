from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import random
from datetime import datetime, timedelta


class TriggerType(Enum):
    PERIODIC = "periodic"
    SELF_DEMAND = "self_demand"
    GITHUB_ISSUE = "github_issue"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class TriggerEvent:
    type: TriggerType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    id: str = ""
    repo_id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.repo_id}_{self.type.value}_{int(self.timestamp)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "priority": self.priority,
            "repo_id": self.repo_id
        }


class TriggerDetector:
    SIMULATED_ISSUES = [
        {"title": "Support max value limit in counter", "body": "The counter should have a configurable maximum value to prevent overflow.", "labels": ["enhancement"]},
        {"title": "Add rate limiting to API", "body": "The API needs rate limiting to prevent abuse and ensure fair usage.", "labels": ["security"]},
        {"title": "Improve error messages", "body": "Current error messages are too cryptic. Need more descriptive messages.", "labels": ["usability"]},
        {"title": "Add health check endpoint", "body": "Need a /health endpoint for Kubernetes liveness/readiness probes.", "labels": ["devops"]},
        {"title": "Support batch operations", "body": "Allow multiple counter operations in a single request.", "labels": ["feature"]},
        {"title": "Fix NullPointerException in API handler", "body": "API returns NPE when request body is null.", "labels": ["bug"]},
        {"title": "Add timeout for HTTP requests", "body": "External API calls should have timeout to prevent hanging.", "labels": ["reliability"]},
        {"title": "Update dependencies", "body": "Security vulnerabilities found in third-party libraries.", "labels": ["security"]},
        {"title": "Add logging middleware", "body": "Need structured logging for all API requests.", "labels": ["observability"]},
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.last_periodic_trigger = time.time()
        self.trigger_handlers: Dict[TriggerType, List[Callable]] = {}
        self.enabled_triggers = self.config.get("enabled", ["periodic", "self_demand"])
        
        self.repos = self.config.get("repos", [])
        if not self.repos:
            self.repos = [{"id": "default", "name": "Default Repo", "github_repo": "", "github_token": ""}]
        
        self.repo_last_trigger: Dict[str, float] = {repo["id"]: time.time() for repo in self.repos}
    
    def register_handler(self, trigger_type: TriggerType, handler: Callable):
        if trigger_type not in self.trigger_handlers:
            self.trigger_handlers[trigger_type] = []
        self.trigger_handlers[trigger_type].append(handler)
    
    def check_triggers(self) -> List[TriggerEvent]:
        events = []
        
        for repo in self.repos:
            repo_id = repo["id"]
            
            if "periodic" in self.enabled_triggers:
                periodic_event = self._check_periodic(repo_id)
                if periodic_event:
                    events.append(periodic_event)
            
            if "self_demand" in self.enabled_triggers:
                self_demand_event = self._check_self_demand(repo_id)
                if self_demand_event:
                    events.append(self_demand_event)
            
            if "github_issue" in self.enabled_triggers:
                github_events = self._check_github_issues(repo)
                events.extend(github_events)
        
        events.sort(key=lambda e: -e.priority)
        
        return events
    
    def _check_periodic(self, repo_id: str) -> Optional[TriggerEvent]:
        interval_minutes = self.config.get("periodic_interval_minutes", 1440)
        interval_seconds = interval_minutes * 60
        
        last_trigger = self.repo_last_trigger.get(repo_id, 0)
        
        if time.time() - last_trigger >= interval_seconds:
            self.repo_last_trigger[repo_id] = time.time()
            return TriggerEvent(
                type=TriggerType.PERIODIC,
                timestamp=time.time(),
                data={"interval_minutes": interval_minutes, "repo_id": repo_id},
                priority=10,
                repo_id=repo_id
            )
        
        return None
    
    def _check_self_demand(self, repo_id: str) -> Optional[TriggerEvent]:
        threshold = self.config.get("self_demand_threshold", 0.7)
        
        demand_score = self._calculate_demand_score(repo_id)
        
        if demand_score >= threshold:
            return TriggerEvent(
                type=TriggerType.SELF_DEMAND,
                timestamp=time.time(),
                data={
                    "demand_score": demand_score,
                    "threshold": threshold,
                    "reason": self._get_demand_reason(demand_score),
                    "repo_id": repo_id
                },
                priority=20,
                repo_id=repo_id
            )
        
        return None
    
    def _calculate_demand_score(self, repo_id: str) -> float:
        factors = []
        
        factors.append(random.uniform(0.3, 0.9))
        factors.append(random.uniform(0.2, 0.8))
        factors.append(random.uniform(0.4, 0.9))
        
        return sum(factors) / len(factors)
    
    def _get_demand_reason(self, score: float) -> str:
        if score >= 0.85:
            return "High technical debt detected, urgent evolution needed"
        elif score >= 0.75:
            return "Multiple improvements identified, ready for evolution"
        else:
            return "Routine self-improvement cycle"
    
    def _check_github_issues(self, repo: Dict[str, Any]) -> List[TriggerEvent]:
        repo_id = repo["id"]
        github_repo = repo.get("github_repo", "")
        github_token = repo.get("github_token", "")
        
        if not github_repo or not github_token:
            return self._generate_simulated_github_events(repo_id)
        
        try:
            return self._fetch_github_issues(github_repo, github_token, repo_id)
        except Exception:
            return self._generate_simulated_github_events(repo_id)
    
    def _fetch_github_issues(self, repo: str, token: str, repo_id: str) -> List[TriggerEvent]:
        events = []
        try:
            from github import Github
            g = Github(token)
            repo_obj = g.get_repo(repo)
            
            for issue in repo_obj.get_issues(state="open")[:5]:
                events.append(TriggerEvent(
                    type=TriggerType.GITHUB_ISSUE,
                    timestamp=time.time(),
                    data={
                        "issue_number": issue.number,
                        "title": issue.title,
                        "body": issue.body or "",
                        "labels": [label.name for label in issue.labels],
                        "url": issue.html_url,
                        "repo_id": repo_id
                    },
                    priority=30,
                    repo_id=repo_id
                ))
        except Exception:
            pass
        
        return events
    
    def _generate_simulated_github_events(self, repo_id: str) -> List[TriggerEvent]:
        events = []
        
        if random.random() < 0.5:
            issue = random.choice(self.SIMULATED_ISSUES)
            events.append(TriggerEvent(
                type=TriggerType.GITHUB_ISSUE,
                timestamp=time.time(),
                data={
                    "issue_number": random.randint(100, 300),
                    "title": issue["title"],
                    "body": issue["body"],
                    "labels": issue["labels"],
                    "url": f"https://github.com/example/{repo_id}/issues/{random.randint(100, 300)}",
                    "repo_id": repo_id
                },
                priority=30,
                repo_id=repo_id
            ))
        
        return events
    
    def create_manual_trigger(self, data: Dict[str, Any] = None, repo_id: str = "") -> TriggerEvent:
        return TriggerEvent(
            type=TriggerType.MANUAL,
            timestamp=time.time(),
            data=data or {},
            priority=50,
            repo_id=repo_id
        )
    
    def create_error_recovery_trigger(self, error_info: Dict[str, Any], repo_id: str = "") -> TriggerEvent:
        return TriggerEvent(
            type=TriggerType.ERROR_RECOVERY,
            timestamp=time.time(),
            data=error_info,
            priority=100,
            repo_id=repo_id
        )
    
    def get_repo_config(self, repo_id: str) -> Optional[Dict[str, Any]]:
        for repo in self.repos:
            if repo["id"] == repo_id:
                return repo
        return None
    
    def get_all_repo_ids(self) -> List[str]:
        return [repo["id"] for repo in self.repos]