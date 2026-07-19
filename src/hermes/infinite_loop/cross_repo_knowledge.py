from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import sqlite3
import os
import re
from difflib import SequenceMatcher
import hashlib


@dataclass
class FixTemplate:
    template_id: str
    source_repo_id: str
    issue_title: str
    issue_labels: List[str]
    diff_content: str
    affected_files: List[str]
    fix_pattern: str
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "source_repo_id": self.source_repo_id,
            "issue_title": self.issue_title,
            "issue_labels": self.issue_labels,
            "diff_content": self.diff_content,
            "affected_files": self.affected_files,
            "fix_pattern": self.fix_pattern,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "created_at": self.created_at
        }


@dataclass
class MatchingResult:
    template: FixTemplate
    target_repo_id: str
    similarity_score: float
    matched_files: List[str]
    recommended_action: str


class CrossRepoKnowledge:
    def __init__(self, db_path: str = "cross_repo_knowledge.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()
    
    def _init_db(self):
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path)
        cursor = self._conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fix_templates (
                template_id TEXT PRIMARY KEY,
                source_repo_id TEXT NOT NULL,
                issue_title TEXT NOT NULL,
                issue_labels TEXT NOT NULL,
                diff_content TEXT NOT NULL,
                affected_files TEXT NOT NULL,
                fix_pattern TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repo_features (
                repo_id TEXT PRIMARY KEY,
                language TEXT,
                file_patterns TEXT,
                dependency_hash TEXT,
                last_updated REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_records (
                migration_id TEXT PRIMARY KEY,
                template_id TEXT NOT NULL,
                source_repo_id TEXT NOT NULL,
                target_repo_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_templates_repo ON fix_templates(source_repo_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_templates_pattern ON fix_templates(fix_pattern)
        ''')
        
        self._conn.commit()
    
    def add_fix_template(self, template: FixTemplate) -> bool:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO fix_templates 
                (template_id, source_repo_id, issue_title, issue_labels, 
                 diff_content, affected_files, fix_pattern, 
                 success_rate, usage_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template.template_id,
                template.source_repo_id,
                template.issue_title,
                ",".join(template.issue_labels),
                template.diff_content,
                ",".join(template.affected_files),
                template.fix_pattern,
                template.success_rate,
                template.usage_count,
                template.created_at
            ))
            self._conn.commit()
            return True
        except Exception:
            return False
    
    def get_template(self, template_id: str) -> Optional[FixTemplate]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT template_id, source_repo_id, issue_title, issue_labels,
                       diff_content, affected_files, fix_pattern, 
                       success_rate, usage_count, created_at
                FROM fix_templates WHERE template_id = ?
            ''', (template_id,))
            
            row = cursor.fetchone()
            if row:
                return FixTemplate(
                    template_id=row[0],
                    source_repo_id=row[1],
                    issue_title=row[2],
                    issue_labels=row[3].split(",") if row[3] else [],
                    diff_content=row[4],
                    affected_files=row[5].split(",") if row[5] else [],
                    fix_pattern=row[6],
                    success_rate=row[7],
                    usage_count=row[8],
                    created_at=row[9]
                )
        except Exception:
            pass
        return None
    
    def get_all_templates(self) -> List[FixTemplate]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT template_id, source_repo_id, issue_title, issue_labels,
                       diff_content, affected_files, fix_pattern, 
                       success_rate, usage_count, created_at
                FROM fix_templates
            ''')
            
            templates = []
            for row in cursor.fetchall():
                templates.append(FixTemplate(
                    template_id=row[0],
                    source_repo_id=row[1],
                    issue_title=row[2],
                    issue_labels=row[3].split(",") if row[3] else [],
                    diff_content=row[4],
                    affected_files=row[5].split(",") if row[5] else [],
                    fix_pattern=row[6],
                    success_rate=row[7],
                    usage_count=row[8],
                    created_at=row[9]
                ))
            return templates
        except Exception:
            return []
    
    def calculate_similarity(self, template: FixTemplate, target_issue: Dict[str, Any]) -> float:
        scores = []
        
        title_sim = SequenceMatcher(
            None, template.issue_title.lower(), 
            target_issue.get("title", "").lower()
        ).ratio()
        scores.append(title_sim * 0.3)
        
        template_labels = set(template.issue_labels)
        target_labels = set(target_issue.get("labels", []))
        if template_labels or target_labels:
            label_sim = len(template_labels & target_labels) / max(len(template_labels), len(target_labels))
            scores.append(label_sim * 0.2)
        
        body_sim = SequenceMatcher(
            None, template.issue_title.lower(), 
            target_issue.get("body", "").lower()
        ).ratio()
        scores.append(body_sim * 0.2)
        
        pattern_sim = self._match_file_patterns(template, target_issue)
        scores.append(pattern_sim * 0.3)
        
        return sum(scores)
    
    def _match_file_patterns(self, template: FixTemplate, target_issue: Dict[str, Any]) -> float:
        target_files = self._extract_file_hints(target_issue)
        if not target_files:
            return 0.5
        
        matched = 0
        for template_file in template.affected_files:
            template_ext = os.path.splitext(template_file)[1]
            for target_file in target_files:
                if template_ext == os.path.splitext(target_file)[1]:
                    matched += 1
                    break
        
        return matched / max(len(template.affected_files), len(target_files))
    
    def _extract_file_hints(self, issue: Dict[str, Any]) -> List[str]:
        body = issue.get("body", "")
        hints = []
        
        patterns = [
            r"`([^`]+\.[a-zA-Z]+)`",
            r"file:\s*([^\s]+)",
            r"([^\s]+\.(py|js|ts|go|rs|java))"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    hints.append(match[0])
                else:
                    hints.append(match)
        
        return list(set(hints))
    
    def find_matching_templates(self, target_issue: Dict[str, Any], 
                               target_repo_id: str,
                               similarity_threshold: float = 0.5) -> List[MatchingResult]:
        all_templates = self.get_all_templates()
        
        results = []
        for template in all_templates:
            if template.source_repo_id == target_repo_id:
                continue
            
            similarity = self.calculate_similarity(template, target_issue)
            if similarity >= similarity_threshold:
                matched_files = self._find_matched_files(template, target_issue)
                
                action = "apply_fix"
                if similarity >= 0.8:
                    action = "apply_fix"
                elif similarity >= 0.6:
                    action = "apply_with_review"
                else:
                    action = "suggest_review"
                
                results.append(MatchingResult(
                    template=template,
                    target_repo_id=target_repo_id,
                    similarity_score=similarity,
                    matched_files=matched_files,
                    recommended_action=action
                ))
        
        results.sort(key=lambda r: -r.similarity_score)
        return results
    
    def _find_matched_files(self, template: FixTemplate, target_issue: Dict[str, Any]) -> List[str]:
        target_files = self._extract_file_hints(target_issue)
        matched = []
        
        for template_file in template.affected_files:
            template_ext = os.path.splitext(template_file)[1]
            for target_file in target_files:
                if template_ext == os.path.splitext(target_file)[1]:
                    matched.append(target_file)
                    break
        
        return matched
    
    def record_migration(self, template_id: str, source_repo_id: str, 
                         target_repo_id: str, similarity_score: float, status: str):
        migration_id = hashlib.md5(f"{template_id}_{target_repo_id}_{status}".encode()).hexdigest()
        
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO migration_records
                (migration_id, template_id, source_repo_id, target_repo_id,
                 similarity_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                migration_id,
                template_id,
                source_repo_id,
                target_repo_id,
                similarity_score,
                status,
                time.time()
            ))
            self._conn.commit()
        except Exception:
            pass
    
    def update_template_success(self, template_id: str, success: bool):
        try:
            template = self.get_template(template_id)
            if template:
                template.usage_count += 1
                if success:
                    template.success_rate = (template.success_rate * (template.usage_count - 1) + 1.0) / template.usage_count
                else:
                    template.success_rate = (template.success_rate * (template.usage_count - 1) + 0.0) / template.usage_count
                
                self.add_fix_template(template)
        except Exception:
            pass


import time


def generate_template_id(source_repo_id: str, issue_number: int) -> str:
    return f"{source_repo_id}_fix_{issue_number}_{int(time.time())}"


def create_fix_template(source_repo_id: str, issue_data: Dict[str, Any], 
                        diff_content: str, affected_files: List[str]) -> FixTemplate:
    fix_pattern = _extract_fix_pattern(diff_content)
    
    issue_number = issue_data.get("issue_number", 0)
    template_id = generate_template_id(source_repo_id, issue_number)
    
    return FixTemplate(
        template_id=template_id,
        source_repo_id=source_repo_id,
        issue_title=issue_data.get("title", ""),
        issue_labels=issue_data.get("labels", []),
        diff_content=diff_content,
        affected_files=affected_files,
        fix_pattern=fix_pattern,
        success_rate=1.0,
        usage_count=1,
        created_at=time.time()
    )


def _extract_fix_pattern(diff_content: str) -> str:
    lines = diff_content.split("\n")
    additions = []
    deletions = []
    
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            deletions.append(line[1:].strip())
    
    if additions:
        return "_".join([a[:20] for a in additions if a][:3])
    return "unknown_pattern"