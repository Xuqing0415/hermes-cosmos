from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

class RuleType(Enum):
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    TRANSFORMATION = "transformation"

class GrammarStatus(Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"

@dataclass
class GrammarRule:
    rule_id: str
    name: str
    rule_type: RuleType
    pattern: str
    description: str = ""
    status: GrammarStatus = GrammarStatus.STABLE
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: float = 0.0
    version: str = "1.0"
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class GrammarSpec:
    name: str
    version: str
    rules: List[GrammarRule] = field(default_factory=list)
    description: str = ""
    
@dataclass
class GrammarFeedback:
    rule_id: str
    test_pass_rate: float
    usage_count: int
    error_patterns: List[str] = field(default_factory=list)
    suggestion: str = ""
