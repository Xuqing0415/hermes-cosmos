from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

class Language(Enum):
    PYTHON = "python"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    C = "c"
    CPP = "cpp"

class MigrationStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"

@dataclass
class MigrationRule:
    rule_id: str
    source_lang: Language
    target_lang: Language
    pattern: str
    translation: str
    description: str = ""
    confidence: float = 1.0
    usage_count: int = 0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class MigrationResult:
    source_code: str
    target_code: str
    source_lang: Language
    target_lang: Language
    status: MigrationStatus
    rules_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
