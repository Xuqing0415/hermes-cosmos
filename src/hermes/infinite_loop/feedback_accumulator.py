from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import os
import time


class FeedbackPhase(Enum):
    REQUIREMENT_PARSING = "requirement_parsing"
    SPEC_GENERATION = "spec_generation"
    CODE_GENERATION = "code_generation"
    DEPLOYMENT = "deployment"
    RUNTIME_MONITORING = "runtime_monitoring"
    AUTO_REPAIR = "auto_repair"
    SELF_EXAMINATION = "self_examination"
    TREND_ANALYSIS = "trend_analysis"
    ROADMAP_GENERATION = "roadmap_generation"
    METACOGNITION = "metacognition"
    SELF_UPGRADE = "self_upgrade"


class FeedbackStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FeedbackEntry:
    phase: FeedbackPhase
    status: FeedbackStatus
    iteration: int
    timestamp: float
    duration: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "data": self.data,
            "message": self.message
        }


@dataclass
class IterationRecord:
    iteration: int
    trigger_type: str
    trigger_data: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    phases: List[FeedbackEntry] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "trigger_type": self.trigger_type,
            "trigger_data": self.trigger_data,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "duration": self.duration,
            "phases": [p.to_dict() for p in self.phases]
        }


class FeedbackAccumulator:
    def __init__(self, db_path: str = "./loop_state.db"):
        self.db_path = db_path
        self._init_db()
        self.current_iteration: Optional[IterationRecord] = None
    
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER UNIQUE,
                trigger_type TEXT,
                trigger_data TEXT,
                start_time REAL,
                end_time REAL,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                phase TEXT,
                status TEXT,
                timestamp REAL,
                duration REAL,
                data TEXT,
                message TEXT,
                FOREIGN KEY (iteration) REFERENCES iterations (iteration)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS objectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                timestamp REAL,
                weights TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_iteration(self, iteration: int, trigger_type: str, trigger_data: Dict[str, Any]):
        self.current_iteration = IterationRecord(
            iteration=iteration,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            start_time=time.time()
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO iterations 
            (iteration, trigger_type, trigger_data, start_time, end_time, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (iteration, trigger_type, json.dumps(trigger_data), time.time(), None, "running"))
        conn.commit()
        conn.close()
    
    def record_feedback(self, phase: FeedbackPhase, status: FeedbackStatus, 
                        data: Dict[str, Any] = None, message: str = "", 
                        duration: float = 0.0):
        if self.current_iteration is None:
            return
        
        entry = FeedbackEntry(
            phase=phase,
            status=status,
            iteration=self.current_iteration.iteration,
            timestamp=time.time(),
            duration=duration,
            data=data or {},
            message=message
        )
        
        self.current_iteration.phases.append(entry)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (iteration, phase, status, timestamp, duration, data, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_iteration.iteration,
            phase.value,
            status.value,
            entry.timestamp,
            entry.duration,
            json.dumps(entry.data),
            entry.message
        ))
        conn.commit()
        conn.close()
    
    def end_iteration(self, status: str = "completed"):
        if self.current_iteration is None:
            return
        
        self.current_iteration.end_time = time.time()
        self.current_iteration.status = status
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE iterations SET end_time = ?, status = ? WHERE iteration = ?
        ''', (self.current_iteration.end_time, status, self.current_iteration.iteration))
        conn.commit()
        conn.close()
    
    def update_objectives(self, weights: Dict[str, float]):
        if self.current_iteration is None:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO objectives (iteration, timestamp, weights)
            VALUES (?, ?, ?)
        ''', (self.current_iteration.iteration, time.time(), json.dumps(weights)))
        conn.commit()
        conn.close()
    
    def get_iteration(self, iteration: int) -> Optional[IterationRecord]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM iterations WHERE iteration = ?', (iteration,))
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            return None
        
        record = IterationRecord(
            iteration=row[1],
            trigger_type=row[2],
            trigger_data=json.loads(row[3]),
            start_time=row[4],
            end_time=row[5],
            status=row[6],
            phases=[]
        )
        
        cursor.execute('SELECT * FROM feedback WHERE iteration = ? ORDER BY timestamp', (iteration,))
        for feedback_row in cursor.fetchall():
            record.phases.append(FeedbackEntry(
                phase=FeedbackPhase(feedback_row[2]),
                status=FeedbackStatus(feedback_row[3]),
                iteration=feedback_row[1],
                timestamp=feedback_row[4],
                duration=feedback_row[5],
                data=json.loads(feedback_row[6]),
                message=feedback_row[7]
            ))
        
        conn.close()
        return record
    
    def get_last_iteration(self) -> Optional[IterationRecord]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(iteration) FROM iterations')
        row = cursor.fetchone()
        
        if row is None or row[0] is None:
            conn.close()
            return None
        
        conn.close()
        return self.get_iteration(row[0])
    
    def get_all_iterations(self) -> List[IterationRecord]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT iteration FROM iterations ORDER BY iteration')
        iterations = []
        
        for row in cursor.fetchall():
            record = self.get_iteration(row[0])
            if record:
                iterations.append(record)
        
        conn.close()
        return iterations
    
    def get_statistics(self) -> Dict[str, Any]:
        iterations = self.get_all_iterations()
        
        if not iterations:
            return {"total_iterations": 0}
        
        completed = [i for i in iterations if i.status == "completed"]
        failed = [i for i in iterations if i.status == "failed"]
        
        avg_duration = sum(i.duration for i in completed) / len(completed) if completed else 0
        
        phase_stats = {}
        for iteration in iterations:
            for phase in iteration.phases:
                if phase.phase.value not in phase_stats:
                    phase_stats[phase.phase.value] = {"success": 0, "failed": 0, "total": 0}
                phase_stats[phase.phase.value]["total"] += 1
                if phase.status == FeedbackStatus.SUCCESS:
                    phase_stats[phase.phase.value]["success"] += 1
                elif phase.status == FeedbackStatus.FAILED:
                    phase_stats[phase.phase.value]["failed"] += 1
        
        return {
            "total_iterations": len(iterations),
            "completed_iterations": len(completed),
            "failed_iterations": len(failed),
            "success_rate": len(completed) / len(iterations),
            "avg_duration": avg_duration,
            "phase_stats": phase_stats
        }
    
    def clear(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM feedback')
        cursor.execute('DELETE FROM iterations')
        cursor.execute('DELETE FROM objectives')
        conn.commit()
        conn.close()