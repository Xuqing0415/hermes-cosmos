import types
from typing import List, Dict, Any, Optional, Type

from .agents import BaseAgent
from .types import AgentRole
from .role_inventor import RoleDefinition


class RoleGenerator:
    _generated_classes: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def generate_agent_class(cls, role_def: RoleDefinition) -> Type[BaseAgent]:
        class_name = role_def.name
        
        if class_name in cls._generated_classes:
            return cls._generated_classes[class_name]
        
        role_value = role_def.name.lower().replace(" ", "_")
        try:
            new_role = AgentRole(role_value)
        except ValueError:
            new_role = AgentRole.RESOURCE_OVERSEER
        
        capabilities = role_def.capabilities
        
        def __init__(self, knowledge_base=None):
            super(self.__class__, self).__init__(
                role=new_role,
                name=class_name,
                capabilities=capabilities,
                knowledge_base=knowledge_base
            )
            self.output_spec = role_def.output_spec
            self.responsibilities = role_def.responsibilities
        
        def execute_task(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
            handler = getattr(self, f"handle_{task_type}", None)
            if handler:
                return handler(input_data)
            
            task_type_clean = task_type.replace("_", "").lower()
            
            for capability in capabilities:
                capability_clean = capability.replace("_", "").lower()
                if capability_clean in task_type_clean or task_type_clean in capability_clean:
                    return self._default_handle(input_data, capability)
            
            for responsibility in self.responsibilities:
                resp_clean = responsibility.replace("_", "").lower()
                if resp_clean in task_type_clean or task_type_clean in resp_clean:
                    return self._default_handle(input_data, responsibility)
            
            return {"error": f"Unknown task type: {task_type}"}
        
        def _default_handle(self, input_data: Dict[str, Any], capability: str) -> Dict[str, Any]:
            result = {"capability": capability, "status": "processed"}
            
            if capability in self.responsibilities:
                result["responsibility"] = capability
            
            for key in self.output_spec.keys():
                if key not in result:
                    result[key] = 0.0 if key == "execution_time" else [] if key.endswith("s") else ""
            
            return result
        
        methods = {
            "__init__": __init__,
            "execute_task": execute_task,
            "_default_handle": _default_handle
        }
        
        for capability in capabilities:
            method_name = f"handle_{capability}"
            
            def create_handler(cap):
                def handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                    return self._default_handle(input_data, cap)
                return handler
            
            methods[method_name] = create_handler(capability)
        
        for responsibility in role_def.responsibilities:
            method_name = f"_{responsibility.lower().replace(' ', '_').replace('.', '')}"
            
            def create_method(desc):
                def method(self, *args, **kwargs):
                    return {"responsibility": desc, "status": "executed"}
                return method
            
            methods[method_name] = create_method(responsibility)
        
        generated_class = type(class_name, (BaseAgent,), methods)
        
        cls._generated_classes[class_name] = generated_class
        
        return generated_class
    
    @classmethod
    def create_agent_instance(cls, role_def: RoleDefinition, knowledge_base=None) -> BaseAgent:
        agent_class = cls.generate_agent_class(role_def)
        return agent_class(knowledge_base)
    
    @classmethod
    def get_generated_class(cls, class_name: str) -> Optional[Type[BaseAgent]]:
        return cls._generated_classes.get(class_name)
    
    @classmethod
    def get_all_generated_classes(cls) -> Dict[str, Type[BaseAgent]]:
        return cls._generated_classes.copy()
    
    @classmethod
    def clear_generated_classes(cls):
        cls._generated_classes.clear()
    
    @classmethod
    def generate_performance_watcher(cls) -> Type[BaseAgent]:
        role_def = RoleDefinition(
            role_id="",
            name="PerformanceWatcher",
            category=None,
            description="Monitors function execution time, identifies performance bottlenecks",
            responsibilities=[
                "run_benchmark",
                "analyze_performance",
                "detect_bottlenecks",
                "suggest_optimizations"
            ],
            capabilities=[
                "performance_analysis",
                "benchmark_testing",
                "bottleneck_detection",
                "optimization_suggestion"
            ],
            input_spec={"source_code": "", "baseline": {}},
            output_spec={"execution_time": 0.0, "bottlenecks": [], "suggestions": []},
            required_skills=["profiling", "benchmarking"]
        )
        
        return cls.generate_agent_class(role_def)