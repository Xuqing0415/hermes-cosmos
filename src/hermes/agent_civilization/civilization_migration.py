import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

from .constitution import Constitution
from .constitution_amendment import ConstitutionAmendmentSystem, Amendment, AmendmentStatus
from .role_inventor import RoleInventor, RoleDefinition, FailurePattern, RoleCategory
from .role_generator import RoleGenerator


class MigrationError(Exception):
    pass


class CivilizationMigration:
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    def export_state(self, include_knowledge: bool = True, include_tasks: bool = True,
                     include_logs: bool = True) -> Dict[str, Any]:
        state = {
            "version": "1.0",
            "export_time": time.time(),
            "export_datetime": datetime.now().isoformat(),
            "civilization": {
                "state": self._export_civilization_state(),
                "constitution": self._export_constitution(),
                "agents": self._export_agents(),
                "amendments": self._export_amendments(),
                "role_inventor": self._export_role_inventor(),
                "generated_roles": self._export_generated_roles()
            }
        }
        
        if include_knowledge:
            state["civilization"]["knowledge"] = self._export_knowledge()
        
        if include_tasks:
            state["civilization"]["tasks"] = self._export_tasks()
        
        if include_logs:
            state["civilization"]["logs"] = self._export_logs()
        
        return state
    
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        try:
            version = state_data.get("version", "1.0")
            
            if version != "1.0":
                raise MigrationError(f"Unsupported version: {version}")
            
            civilization_data = state_data.get("civilization", {})
            
            self._import_constitution(civilization_data.get("constitution", {}))
            self._import_agents(civilization_data.get("agents", []))
            self._import_role_inventor(civilization_data.get("role_inventor", {}))
            self._import_generated_roles(civilization_data.get("generated_roles", {}))
            self._import_amendments(civilization_data.get("amendments", {}))
            
            if "knowledge" in civilization_data:
                self._import_knowledge(civilization_data["knowledge"])
            
            if "tasks" in civilization_data:
                self._import_tasks(civilization_data["tasks"])
            
            if "logs" in civilization_data:
                self._import_logs(civilization_data["logs"])
            
            return True
        
        except Exception as e:
            raise MigrationError(f"Failed to import state: {str(e)}")
    
    def export_to_file(self, filepath: str, **kwargs) -> bool:
        try:
            state = self.export_state(**kwargs)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        
        except Exception as e:
            raise MigrationError(f"Failed to export to file: {str(e)}")
    
    def import_from_file(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            return self.import_state(state_data)
        
        except Exception as e:
            raise MigrationError(f"Failed to import from file: {str(e)}")
    
    def _export_civilization_state(self) -> Dict[str, Any]:
        state = self.coordinator.state
        return {
            "phase": state.phase.value,
            "total_agents": state.total_agents,
            "total_tasks": state.total_tasks,
            "total_knowledge": state.total_knowledge,
            "total_logs": state.total_logs,
            "creation_time": state.creation_time,
            "last_update": state.last_update
        }
    
    def _export_constitution(self) -> Dict[str, Any]:
        return self.coordinator.constitution.to_dict()
    
    def _import_constitution(self, data: Dict[str, Any]):
        if data:
            self.coordinator.constitution = Constitution.from_dict(data)
    
    def _export_agents(self) -> List[Dict[str, Any]]:
        agents = []
        for agent in self.coordinator.get_all_agents():
            agents.append({
                "agent_id": agent.agent_id,
                "role": agent.role.value,
                "name": agent.name,
                "capabilities": agent.capabilities,
                "status": agent.status,
                "last_heartbeat": agent.last_heartbeat,
                "performance_score": agent.performance_score,
                "task_count": agent.task_count,
                "success_rate": agent.success_rate,
                "resources": agent.resources
            })
        return agents
    
    def _import_agents(self, agents_data: List[Dict[str, Any]]):
        from .types import AgentRole, AgentInfo
        
        for agent_data in agents_data:
            agent_info = AgentInfo(
                agent_id=agent_data["agent_id"],
                role=AgentRole(agent_data["role"]),
                name=agent_data["name"],
                capabilities=agent_data["capabilities"],
                status=agent_data.get("status", "active"),
                last_heartbeat=agent_data.get("last_heartbeat", 0.0),
                performance_score=agent_data.get("performance_score", 0.0),
                task_count=agent_data.get("task_count", 0),
                success_rate=agent_data.get("success_rate", 0.0),
                resources=agent_data.get("resources", {})
            )
            self.coordinator.register_agent(agent_info)
    
    def _export_knowledge(self) -> List[Dict[str, Any]]:
        from .types import KnowledgeItem
        
        knowledge_list = []
        for knowledge in self.coordinator.knowledge_base.search_knowledge("", max_results=1000):
            knowledge_list.append({
                "knowledge_id": knowledge.knowledge_id,
                "type": knowledge.type,
                "content": knowledge.content,
                "source_agent_id": knowledge.source_agent_id,
                "source_role": knowledge.source_role.value,
                "created_at": knowledge.created_at,
                "usage_count": knowledge.usage_count,
                "rating": knowledge.rating
            })
        return knowledge_list
    
    def _import_knowledge(self, knowledge_data: List[Dict[str, Any]]):
        from .types import KnowledgeItem, AgentRole
        
        for data in knowledge_data:
            knowledge = KnowledgeItem(
                knowledge_id=data["knowledge_id"],
                type=data["type"],
                content=data["content"],
                source_agent_id=data["source_agent_id"],
                source_role=AgentRole(data["source_role"]),
                created_at=data.get("created_at", time.time()),
                usage_count=data.get("usage_count", 0),
                rating=data.get("rating", 0.0)
            )
            self.coordinator.add_knowledge(knowledge)
    
    def _export_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for task in self.coordinator.task_board.tasks.values():
            tasks.append({
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "type": task.type,
                "input_data": task.input_data,
                "output_data": task.output_data,
                "status": task.status.value,
                "priority": task.priority,
                "assignee": task.assignee,
                "bids": task.bids,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "deadline": task.deadline,
                "dependencies": task.dependencies,
                "max_retries": task.max_retries,
                "retry_count": task.retry_count
            })
        return tasks
    
    def _import_tasks(self, tasks_data: List[Dict[str, Any]]):
        from .types import Task, TaskStatus
        
        for data in tasks_data:
            task = Task(
                task_id=data["task_id"],
                title=data["title"],
                description=data["description"],
                type=data["type"],
                input_data=data.get("input_data", {}),
                output_data=data.get("output_data", {}),
                status=TaskStatus(data.get("status", "pending")),
                priority=data.get("priority", 0),
                assignee=data.get("assignee"),
                bids=data.get("bids", []),
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
                deadline=data.get("deadline"),
                dependencies=data.get("dependencies", []),
                max_retries=data.get("max_retries", 3),
                retry_count=data.get("retry_count", 0)
            )
            self.coordinator.task_board.tasks[task.task_id] = task
            
            status = task.status
            if status == TaskStatus.PENDING:
                self.coordinator.task_board.pending_tasks.append(task.task_id)
            elif status == TaskStatus.IN_PROGRESS:
                self.coordinator.task_board.in_progress_tasks.append(task.task_id)
            elif status == TaskStatus.COMPLETED:
                self.coordinator.task_board.completed_tasks.append(task.task_id)
    
    def _export_logs(self) -> List[Dict[str, Any]]:
        logs = []
        for log in self.coordinator.get_logs(1000):
            logs.append({
                "log_id": log.log_id,
                "timestamp": log.timestamp,
                "agent_id": log.agent_id,
                "agent_role": log.agent_role.value,
                "action": log.action,
                "details": log.details
            })
        return logs
    
    def _import_logs(self, logs_data: List[Dict[str, Any]]):
        from .types import CivilizationLog, AgentRole
        
        for data in logs_data:
            log = CivilizationLog(
                log_id=data["log_id"],
                timestamp=data.get("timestamp", time.time()),
                agent_id=data["agent_id"],
                agent_role=AgentRole(data["agent_role"]),
                action=data["action"],
                details=data.get("details", {})
            )
            self.coordinator.logs.append(log)
    
    def _export_amendments(self) -> Dict[str, Any]:
        return {
            "total_amendments": len(self.coordinator.amendment_system.amendments) if hasattr(self.coordinator, 'amendment_system') else 0
        }
    
    def _import_amendments(self, data: Dict[str, Any]):
        pass
    
    def _export_role_inventor(self) -> Dict[str, Any]:
        return {
            "total_role_definitions": 0,
            "total_failure_patterns": 0
        }
    
    def _import_role_inventor(self, data: Dict[str, Any]):
        pass
    
    def _export_generated_roles(self) -> List[Dict[str, Any]]:
        generated_classes = RoleGenerator.get_all_generated_classes()
        return [{"class_name": name} for name in generated_classes.keys()]
    
    def _import_generated_roles(self, data: List[Dict[str, Any]]):
        for item in data:
            class_name = item.get("class_name")
            if class_name == "PerformanceWatcher":
                RoleGenerator.generate_performance_watcher()
    
    def get_state_summary(self) -> Dict[str, Any]:
        state = self.export_state(include_knowledge=False, include_tasks=False, include_logs=False)
        return {
            "export_time": state["export_datetime"],
            "phase": state["civilization"]["state"]["phase"],
            "total_agents": state["civilization"]["state"]["total_agents"],
            "total_tasks": state["civilization"]["state"]["total_tasks"],
            "total_knowledge": state["civilization"]["state"]["total_knowledge"],
            "constitution_version": state["civilization"]["constitution"]["version"],
            "amendment_count": state["civilization"]["constitution"]["amendment_count"],
            "agent_count": len(state["civilization"]["agents"])
        }