"""
Type definitions for CI/CD proof pipeline
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class ProofResultStatus(Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    CACHED = "cached"


@dataclass
class ChangedFile:
    """A file that was changed in a PR"""
    
    filename: str
    additions: int
    deletions: int
    changes: int
    status: str


@dataclass
class ChangedFunction:
    """A function that was changed in a PR"""
    
    name: str
    filename: str
    line_start: int
    line_end: int
    signature: str
    affected_dependencies: List[str] = None


@dataclass
class ProofCacheEntry:
    """An entry in the proof cache"""
    
    key: str
    function_name: str
    filename: str
    signature: str
    status: ProofResultStatus
    certificate_path: Optional[str] = None
    duration: float = 0.0
    timestamp: Optional[datetime] = None
    model_version: Optional[str] = None


@dataclass
class ProofResult:
    """Result of a proof attempt in CI"""
    
    function_name: str
    filename: str
    status: ProofResultStatus
    duration: float = 0.0
    certificate_path: Optional[str] = None
    generated_tests: List[str] = None
    error_message: Optional[str] = None
    cached: bool = False


@dataclass
class CIProofReport:
    """CI proof report for a PR"""
    
    pr_number: int
    commit_hash: str
    total_proven: int = 0
    total_disproven: int = 0
    total_timeout: int = 0
    total_unknown: int = 0
    total_cached: int = 0
    results: List[ProofResult] = None
    total_duration: float = 0.0
    cache_hit_rate: float = 0.0
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


@dataclass
class PRInfo:
    """Information about a GitHub PR"""
    
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    base_sha: str
    title: str
    author: str


@dataclass
class DependencyGraph:
    """Function dependency graph"""
    
    functions: List[str]
    dependencies: Dict[str, List[str]]
    
    def get_affected_functions(self, changed_functions: List[str]) -> List[str]:
        """Get all functions affected by changes"""
        affected = set(changed_functions)
        
        changed_set = set(changed_functions)
        changed_this_iteration = set(changed_functions)
        
        while changed_this_iteration:
            next_changed = set()
            for func, deps in self.dependencies.items():
                if func not in affected and any(d in changed_this_iteration for d in deps):
                    affected.add(func)
                    next_changed.add(func)
            changed_this_iteration = next_changed
        
        return list(affected)
