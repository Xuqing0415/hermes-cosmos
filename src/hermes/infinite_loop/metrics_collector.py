from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import sqlite3
import os
import time


@dataclass
class MigrationMetric:
    migration_id: str
    template_id: str
    source_repo_id: str
    target_repo_id: str
    similarity_score: float
    success: bool
    test_pass_rate: float
    pr_created: bool
    created_at: float


@dataclass
class RepoActivity:
    repo_id: str
    active_issues: int
    fixes_applied: int
    migrations_received: int
    migrations_sent: int
    last_activity: float


class MetricsCollector:
    def __init__(self, db_path: str = "cross_repo_metrics.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()
    
    def _init_db(self):
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path)
        cursor = self._conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_metrics (
                migration_id TEXT PRIMARY KEY,
                template_id TEXT NOT NULL,
                source_repo_id TEXT NOT NULL,
                target_repo_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                success INTEGER NOT NULL,
                test_pass_rate REAL DEFAULT 0.0,
                pr_created INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repo_activity (
                repo_id TEXT PRIMARY KEY,
                active_issues INTEGER DEFAULT 0,
                fixes_applied INTEGER DEFAULT 0,
                migrations_received INTEGER DEFAULT 0,
                migrations_sent INTEGER DEFAULT 0,
                last_activity REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threshold_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                old_threshold REAL NOT NULL,
                new_threshold REAL NOT NULL,
                reason TEXT NOT NULL,
                success_rate REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_source ON migration_metrics(source_repo_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_target ON migration_metrics(target_repo_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metrics_success ON migration_metrics(success)
        ''')
        
        self._conn.commit()
    
    def record_migration(self, metric: MigrationMetric):
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO migration_metrics
                (migration_id, template_id, source_repo_id, target_repo_id,
                 similarity_score, success, test_pass_rate, pr_created, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.migration_id,
                metric.template_id,
                metric.source_repo_id,
                metric.target_repo_id,
                metric.similarity_score,
                1 if metric.success else 0,
                metric.test_pass_rate,
                1 if metric.pr_created else 0,
                metric.created_at
            ))
            self._conn.commit()
        except Exception:
            pass
    
    def update_repo_activity(self, repo_id: str, **kwargs):
        try:
            cursor = self._conn.cursor()
            
            cursor.execute('''
                SELECT active_issues, fixes_applied, migrations_received, migrations_sent
                FROM repo_activity WHERE repo_id = ?
            ''', (repo_id,))
            
            row = cursor.fetchone()
            if row:
                active_issues = kwargs.get('active_issues', row[0])
                fixes_applied = kwargs.get('fixes_applied', row[1])
                migrations_received = kwargs.get('migrations_received', row[2])
                migrations_sent = kwargs.get('migrations_sent', row[3])
            else:
                active_issues = kwargs.get('active_issues', 0)
                fixes_applied = kwargs.get('fixes_applied', 0)
                migrations_received = kwargs.get('migrations_received', 0)
                migrations_sent = kwargs.get('migrations_sent', 0)
            
            cursor.execute('''
                INSERT OR REPLACE INTO repo_activity
                (repo_id, active_issues, fixes_applied, migrations_received, migrations_sent, last_activity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                repo_id,
                active_issues,
                fixes_applied,
                migrations_received,
                migrations_sent,
                time.time()
            ))
            self._conn.commit()
        except Exception:
            pass
    
    def get_migration_success_rate(self, source_repo_id: str = None, 
                                   target_repo_id: str = None,
                                   min_similarity: float = 0.0) -> float:
        try:
            cursor = self._conn.cursor()
            
            query = '''
                SELECT COUNT(*) as total, SUM(success) as successes
                FROM migration_metrics
                WHERE similarity_score >= ?
            '''
            params = [min_similarity]
            
            if source_repo_id:
                query += " AND source_repo_id = ?"
                params.append(source_repo_id)
            
            if target_repo_id:
                query += " AND target_repo_id = ?"
                params.append(target_repo_id)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return row[1] / row[0]
        except Exception:
            pass
        
        return 0.5
    
    def get_repo_activity(self, repo_id: str) -> Optional[RepoActivity]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT repo_id, active_issues, fixes_applied, migrations_received, migrations_sent, last_activity
                FROM repo_activity WHERE repo_id = ?
            ''', (repo_id,))
            
            row = cursor.fetchone()
            if row:
                return RepoActivity(
                    repo_id=row[0],
                    active_issues=row[1],
                    fixes_applied=row[2],
                    migrations_received=row[3],
                    migrations_sent=row[4],
                    last_activity=row[5]
                )
        except Exception:
            pass
        return None
    
    def get_all_repo_activities(self) -> List[RepoActivity]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT repo_id, active_issues, fixes_applied, migrations_received, migrations_sent, last_activity
                FROM repo_activity
            ''')
            
            activities = []
            for row in cursor.fetchall():
                activities.append(RepoActivity(
                    repo_id=row[0],
                    active_issues=row[1],
                    fixes_applied=row[2],
                    migrations_received=row[3],
                    migrations_sent=row[4],
                    last_activity=row[5]
                ))
            return activities
        except Exception:
            return []
    
    def suggest_similarity_threshold(self) -> float:
        success_rates = []
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for threshold in thresholds:
            rate = self.get_migration_success_rate(min_similarity=threshold)
            success_rates.append((threshold, rate))
        
        for threshold, rate in success_rates:
            if rate >= 0.7:
                return threshold
        
        return 0.5
    
    def record_threshold_adjustment(self, old_threshold: float, new_threshold: float, 
                                    reason: str, success_rate: float):
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT INTO threshold_adjustments
                (timestamp, old_threshold, new_threshold, reason, success_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                time.time(),
                old_threshold,
                new_threshold,
                reason,
                success_rate
            ))
            self._conn.commit()
        except Exception:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        try:
            cursor = self._conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM migration_metrics')
            total_migrations = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(success) FROM migration_metrics')
            successful_migrations = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(DISTINCT source_repo_id) FROM migration_metrics')
            active_source_repos = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT target_repo_id) FROM migration_metrics')
            active_target_repos = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(similarity_score) FROM migration_metrics')
            avg_similarity = cursor.fetchone()[0] or 0
            
            return {
                "total_migrations": total_migrations,
                "successful_migrations": successful_migrations,
                "success_rate": successful_migrations / total_migrations if total_migrations > 0 else 0,
                "active_source_repos": active_source_repos,
                "active_target_repos": active_target_repos,
                "avg_similarity": avg_similarity
            }
        except Exception:
            return {
                "total_migrations": 0,
                "successful_migrations": 0,
                "success_rate": 0,
                "active_source_repos": 0,
                "active_target_repos": 0,
                "avg_similarity": 0
            }