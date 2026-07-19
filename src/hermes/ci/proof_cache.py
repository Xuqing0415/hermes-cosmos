"""
Proof Cache - Stores proven signatures for incremental verification
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import structlog

from hermes.ci.types import ProofCacheEntry, ProofResultStatus

logger = structlog.get_logger()


class ProofCache:
    """
    Cache for storing and retrieving proof results.
    
    Uses SQLite as the backend (no external dependencies needed).
    Supports:
    - Storing proof results by function signature
    - Retrieving cached results
    - Invalidating stale entries
    - Computing cache statistics
    """
    
    def __init__(self, cache_path: str = "./proof_cache.db"):
        self.cache_path = cache_path
        self._conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        if self.cache_path != ":memory:":
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        self._conn = sqlite3.connect(self.cache_path)
        cursor = self._conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proof_cache (
                key TEXT PRIMARY KEY,
                function_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                signature TEXT NOT NULL,
                status TEXT NOT NULL,
                certificate_path TEXT,
                duration REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_function_name ON proof_cache(function_name)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_filename ON proof_cache(filename)
        ''')
        
        self._conn.commit()
    
    def compute_key(self, function_name: str, source_code: str) -> str:
        """
        Compute a cache key from function name and source code.
        
        Args:
            function_name: Name of the function
            source_code: Source code of the function
        
        Returns:
            Cache key string
        """
        content = f"{function_name}:{source_code}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[ProofCacheEntry]:
        """
        Retrieve a cached proof result.
        
        Args:
            key: Cache key
        
        Returns:
            ProofCacheEntry or None
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM proof_cache WHERE key = ?",
            (key,)
        )
        
        row = cursor.fetchone()
        if row:
            return ProofCacheEntry(
                key=row[0],
                function_name=row[1],
                filename=row[2],
                signature=row[3],
                status=ProofResultStatus(row[4]),
                certificate_path=row[5],
                duration=row[6],
                timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                model_version=row[8]
            )
        
        return None
    
    def set(
        self,
        key: str,
        function_name: str,
        filename: str,
        signature: str,
        status: ProofResultStatus,
        certificate_path: Optional[str] = None,
        duration: float = 0.0,
        model_version: Optional[str] = None
    ):
        """
        Store a proof result in the cache.
        
        Args:
            key: Cache key
            function_name: Name of the function
            filename: File containing the function
            signature: Function signature hash
            status: Proof result status
            certificate_path: Path to proof certificate
            duration: Proof duration in seconds
            model_version: Version of the theorem prover model
        """
        cursor = self._conn.cursor()
        cursor.execute('''
            REPLACE INTO proof_cache (
                key, function_name, filename, signature,
                status, certificate_path, duration, timestamp, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            key, function_name, filename, signature,
            status.value, certificate_path, duration,
            datetime.now().isoformat(), model_version
        ))
        
        self._conn.commit()
        logger.info("Proof cached", key=key, function_name=function_name, status=status.value)
    
    def invalidate(self, function_name: str):
        """
        Invalidate all cache entries for a function.
        
        Args:
            function_name: Name of the function to invalidate
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM proof_cache WHERE function_name = ?",
            (function_name,)
        )
        self._conn.commit()
        logger.info("Cache invalidated", function_name=function_name)
    
    def invalidate_file(self, filename: str):
        """
        Invalidate all cache entries for a file.
        
        Args:
            filename: Path to the file to invalidate
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM proof_cache WHERE filename = ?",
            (filename,)
        )
        self._conn.commit()
        logger.info("Cache invalidated for file", filename=filename)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        cursor = self._conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM proof_cache")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM proof_cache WHERE status = 'proven'")
        proven = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM proof_cache WHERE status = 'disproven'")
        disproven = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM proof_cache WHERE status = 'timeout'")
        timeout = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(duration) FROM proof_cache")
        avg_duration = cursor.fetchone()[0] or 0.0
        
        return {
            "total_entries": total,
            "proven": proven,
            "disproven": disproven,
            "timeout": timeout,
            "avg_duration": round(avg_duration, 2),
            "hit_rate": 0.0
        }
    
    def get_entries_for_function(self, function_name: str) -> List[ProofCacheEntry]:
        """
        Get all cache entries for a function.
        
        Args:
            function_name: Name of the function
        
        Returns:
            List of ProofCacheEntry
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM proof_cache WHERE function_name = ?",
            (function_name,)
        )
        
        entries = []
        for row in cursor.fetchall():
            entries.append(ProofCacheEntry(
                key=row[0],
                function_name=row[1],
                filename=row[2],
                signature=row[3],
                status=ProofResultStatus(row[4]),
                certificate_path=row[5],
                duration=row[6],
                timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                model_version=row[8]
            ))
        
        return entries
    
    def purge_old_entries(self, days: int = 30):
        """
        Purge entries older than a certain number of days.
        
        Args:
            days: Maximum age in days
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM proof_cache WHERE timestamp < ?",
            (cutoff.isoformat(),)
        )
        
        deleted = cursor.rowcount
        self._conn.commit()
        logger.info("Old entries purged", deleted=deleted)
    
    def close(self):
        """Close the database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache to dictionary for serialization"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM proof_cache")
        
        entries = []
        for row in cursor.fetchall():
            entries.append({
                "key": row[0],
                "function_name": row[1],
                "filename": row[2],
                "signature": row[3],
                "status": row[4],
                "certificate_path": row[5],
                "duration": row[6],
                "timestamp": row[7],
                "model_version": row[8]
            })
        
        return {
            "entries": entries,
            "stats": self.get_stats()
        }
    
    def save_to_file(self, filepath: str):
        """Save cache to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "ProofCache":
        """Load cache from a JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        cache = cls(":memory:")
        
        for entry in data.get("entries", []):
            cache.set(
                key=entry["key"],
                function_name=entry["function_name"],
                filename=entry["filename"],
                signature=entry["signature"],
                status=ProofResultStatus(entry["status"]),
                certificate_path=entry.get("certificate_path"),
                duration=entry.get("duration", 0.0),
                model_version=entry.get("model_version")
            )
        
        return cache
