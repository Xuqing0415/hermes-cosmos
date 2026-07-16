import subprocess
import os
import shutil
import json
import time
from typing import List, Dict, Optional

from .types import RefactorPlan, RefactorResult, RefactorHistoryEntry


class RefactorRollback:
    def __init__(self, backup_dir: str = ".refactor_backups"):
        self.backup_dir = backup_dir
        self.history: List[RefactorHistoryEntry] = []
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

    def create_backup(self, plan: RefactorPlan) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"{plan.plan_id}_{timestamp}")
        
        os.makedirs(backup_path)
        
        all_files = set()
        for action in plan.actions:
            all_files.update(action.affected_files)
        
        for file_path in all_files:
            if os.path.exists(file_path):
                rel_path = os.path.relpath(file_path)
                backup_file = os.path.join(backup_path, rel_path)
                backup_file_dir = os.path.dirname(backup_file)
                
                if not os.path.exists(backup_file_dir):
                    os.makedirs(backup_file_dir)
                
                shutil.copy2(file_path, backup_file)
        
        metadata = {
            'plan_id': plan.plan_id,
            'timestamp': timestamp,
            'affected_files': list(all_files),
            'smell_type': plan.smell.smell_type.value
        }
        
        with open(os.path.join(backup_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return backup_path

    def rollback(self, plan_id: str) -> bool:
        backup_paths = []
        
        for item in os.listdir(self.backup_dir):
            if item.startswith(plan_id):
                backup_paths.append(os.path.join(self.backup_dir, item))
        
        if not backup_paths:
            return False
        
        latest_backup = sorted(backup_paths)[-1]
        
        with open(os.path.join(latest_backup, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for file_path in metadata.get('affected_files', []):
            rel_path = os.path.relpath(file_path)
            backup_file = os.path.join(latest_backup, rel_path)
            
            if os.path.exists(backup_file):
                file_dir = os.path.dirname(file_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                shutil.copy2(backup_file, file_path)
            else:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        self._add_history_entry(
            plan_id=plan_id,
            smell_type=metadata.get('smell_type', 'unknown'),
            actions_taken=['rollback'],
            success=True,
            verification_result='rollback',
            execution_time=0,
            rollback_used=True
        )
        
        return True

    def restore_file(self, plan_id: str, file_path: str) -> bool:
        backup_paths = []
        
        for item in os.listdir(self.backup_dir):
            if item.startswith(plan_id):
                backup_paths.append(os.path.join(self.backup_dir, item))
        
        if not backup_paths:
            return False
        
        latest_backup = sorted(backup_paths)[-1]
        rel_path = os.path.relpath(file_path)
        backup_file = os.path.join(latest_backup, rel_path)
        
        if os.path.exists(backup_file):
            file_dir = os.path.dirname(file_path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            shutil.copy2(backup_file, file_path)
            return True
        
        return False

    def get_backup_info(self, plan_id: str) -> Optional[Dict]:
        backup_paths = []
        
        for item in os.listdir(self.backup_dir):
            if item.startswith(plan_id):
                backup_paths.append(os.path.join(self.backup_dir, item))
        
        if not backup_paths:
            return None
        
        latest_backup = sorted(backup_paths)[-1]
        
        with open(os.path.join(latest_backup, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata

    def list_backups(self) -> List[Dict]:
        backups = []
        
        for item in os.listdir(self.backup_dir):
            item_path = os.path.join(self.backup_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    backups.append(metadata)
        
        return backups

    def _add_history_entry(self, **kwargs):
        entry = RefactorHistoryEntry(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        )
        self.history.append(entry)
        
        history_file = os.path.join(self.backup_dir, 'history.json')
        history_data = []
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        
        history_data.append({
            'timestamp': entry.timestamp,
            'plan_id': entry.plan_id,
            'smell_type': entry.smell_type,
            'actions_taken': entry.actions_taken,
            'success': entry.success,
            'verification_result': entry.verification_result,
            'execution_time': entry.execution_time,
            'rollback_used': entry.rollback_used,
            'notes': entry.notes
        })
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)

    def get_history(self) -> List[RefactorHistoryEntry]:
        history_file = os.path.join(self.backup_dir, 'history.json')
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.history = []
            for item in history_data:
                self.history.append(RefactorHistoryEntry(
                    timestamp=item.get('timestamp', ''),
                    plan_id=item.get('plan_id', ''),
                    smell_type=item.get('smell_type', ''),
                    actions_taken=item.get('actions_taken', []),
                    success=item.get('success', False),
                    verification_result=item.get('verification_result', ''),
                    execution_time=item.get('execution_time', 0),
                    rollback_used=item.get('rollback_used', False),
                    notes=item.get('notes', '')
                ))
        
        return self.history

    def cleanup_old_backups(self, days_to_keep: int = 7):
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for item in os.listdir(self.backup_dir):
            item_path = os.path.join(self.backup_dir, item)
            if os.path.isdir(item_path) and item != 'history':
                try:
                    timestamp_str = item.split('_')[1]
                    timestamp = time.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    item_time = time.mktime(timestamp)
                    
                    if item_time < cutoff_time:
                        shutil.rmtree(item_path)
                except (IndexError, ValueError):
                    pass