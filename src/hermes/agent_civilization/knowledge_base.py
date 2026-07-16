import time
import sqlite3
import os
from typing import List, Dict, Optional, Any

from .types import KnowledgeItem, AgentRole


class KnowledgeBase:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path)
        cursor = self._conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                knowledge_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_agent_id TEXT NOT NULL,
                source_role TEXT NOT NULL,
                created_at REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_knowledge_role ON knowledge(source_role)
        ''')
        
        self._conn.commit()

    def add_knowledge(self, knowledge: KnowledgeItem) -> bool:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge 
                (knowledge_id, type, content, source_agent_id, source_role, 
                 created_at, usage_count, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                knowledge.knowledge_id,
                knowledge.type,
                str(knowledge.content),
                knowledge.source_agent_id,
                knowledge.source_role.value,
                knowledge.created_at,
                knowledge.usage_count,
                knowledge.rating
            ))
            self._conn.commit()
            return True
        except Exception:
            return False

    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT knowledge_id, type, content, source_agent_id, source_role,
                       created_at, usage_count, rating
                FROM knowledge WHERE knowledge_id = ?
            ''', (knowledge_id,))
            
            row = cursor.fetchone()
            if row:
                return KnowledgeItem(
                    knowledge_id=row[0],
                    type=row[1],
                    content=eval(row[2]),
                    source_agent_id=row[3],
                    source_role=AgentRole(row[4]),
                    created_at=row[5],
                    usage_count=row[6],
                    rating=row[7]
                )
            return None
        except Exception:
            return None

    def search_knowledge(self, query: str, knowledge_type: Optional[str] = None,
                         max_results: int = 10) -> List[KnowledgeItem]:
        try:
            cursor = self._conn.cursor()
            
            if knowledge_type:
                cursor.execute('''
                    SELECT knowledge_id, type, content, source_agent_id, source_role,
                           created_at, usage_count, rating
                    FROM knowledge 
                    WHERE type = ? AND content LIKE ?
                    ORDER BY usage_count DESC, rating DESC
                    LIMIT ?
                ''', (knowledge_type, f"%{query}%", max_results))
            else:
                cursor.execute('''
                    SELECT knowledge_id, type, content, source_agent_id, source_role,
                           created_at, usage_count, rating
                    FROM knowledge 
                    WHERE content LIKE ?
                    ORDER BY usage_count DESC, rating DESC
                    LIMIT ?
                ''', (f"%{query}%", max_results))
            
            results = []
            for row in cursor.fetchall():
                results.append(KnowledgeItem(
                    knowledge_id=row[0],
                    type=row[1],
                    content=eval(row[2]),
                    source_agent_id=row[3],
                    source_role=AgentRole(row[4]),
                    created_at=row[5],
                    usage_count=row[6],
                    rating=row[7]
                ))
            
            for item in results:
                self._increment_usage(item.knowledge_id)
            
            return results
        except Exception:
            return []

    def get_by_type(self, knowledge_type: str, max_results: int = 10) -> List[KnowledgeItem]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT knowledge_id, type, content, source_agent_id, source_role,
                       created_at, usage_count, rating
                FROM knowledge 
                WHERE type = ?
                ORDER BY usage_count DESC, rating DESC
                LIMIT ?
            ''', (knowledge_type, max_results))
            
            results = []
            for row in cursor.fetchall():
                results.append(KnowledgeItem(
                    knowledge_id=row[0],
                    type=row[1],
                    content=eval(row[2]),
                    source_agent_id=row[3],
                    source_role=AgentRole(row[4]),
                    created_at=row[5],
                    usage_count=row[6],
                    rating=row[7]
                ))
            
            return results
        except Exception:
            return []

    def get_by_source(self, source_agent_id: str, max_results: int = 10) -> List[KnowledgeItem]:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT knowledge_id, type, content, source_agent_id, source_role,
                       created_at, usage_count, rating
                FROM knowledge 
                WHERE source_agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (source_agent_id, max_results))
            
            results = []
            for row in cursor.fetchall():
                results.append(KnowledgeItem(
                    knowledge_id=row[0],
                    type=row[1],
                    content=eval(row[2]),
                    source_agent_id=row[3],
                    source_role=AgentRole(row[4]),
                    created_at=row[5],
                    usage_count=row[6],
                    rating=row[7]
                ))
            
            return results
        except Exception:
            return []

    def rate_knowledge(self, knowledge_id: str, rating: float) -> bool:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                UPDATE knowledge SET rating = ? WHERE knowledge_id = ?
            ''', (rating, knowledge_id))
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    def _increment_usage(self, knowledge_id: str):
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                UPDATE knowledge SET usage_count = usage_count + 1 WHERE knowledge_id = ?
            ''', (knowledge_id,))
            self._conn.commit()
        except Exception:
            pass

    def delete_knowledge(self, knowledge_id: str) -> bool:
        try:
            cursor = self._conn.cursor()
            cursor.execute('''
                DELETE FROM knowledge WHERE knowledge_id = ?
            ''', (knowledge_id,))
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            cursor = self._conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM knowledge')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT type, COUNT(*) FROM knowledge GROUP BY type')
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute('SELECT AVG(rating) FROM knowledge')
            avg_rating = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT SUM(usage_count) FROM knowledge')
            total_usage = cursor.fetchone()[0] or 0
            
            return {
                "total_knowledge": total,
                "knowledge_by_type": by_type,
                "average_rating": round(avg_rating, 2),
                "total_usage": total_usage
            }
        except Exception:
            return {"total_knowledge": 0, "knowledge_by_type": {}, "average_rating": 0.0, "total_usage": 0}

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None