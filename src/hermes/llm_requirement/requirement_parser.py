import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class EntityType(Enum):
    SERVICE = "service"
    DATABASE = "database"
    COUNTER = "counter"
    STACK = "stack"
    QUEUE = "queue"
    CACHE = "cache"
    API = "api"
    RESOURCE = "resource"
    OTHER = "other"


class OperationType(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    GET = "get"
    SET = "set"
    PUSH = "push"
    POP = "pop"
    RESET = "reset"
    ENQUEUE = "enqueue"
    DEQUEUE = "dequeue"
    CLEAR = "clear"


class ConstraintType(Enum):
    LINEARIZABILITY = "linearizability"
    CONSISTENCY = "consistency"
    ATOMICITY = "atomicity"
    LOCKING = "locking"
    THREAD_SAFETY = "thread_safety"
    NO_LOST_UPDATES = "no_lost_updates"
    ORDER_PRESERVING = "order_preserving"
    FAIRNESS = "fairness"
    LIVENESS = "liveness"
    SAFETY = "safety"


@dataclass
class Entity:
    name: str
    entity_type: EntityType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Operation:
    name: str
    operation_type: OperationType
    endpoint: str = ""
    method: str = "POST"
    parameters: List[str] = field(default_factory=list)
    returns: str = ""
    description: str = ""


@dataclass
class Constraint:
    name: str
    constraint_type: ConstraintType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRequirement:
    name: str
    description: str
    entities: List[Entity] = field(default_factory=list)
    operations: List[Operation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    language: str = "python"
    framework: str = "flask"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entities": [{
                "name": e.name,
                "entity_type": e.entity_type.value,
                "description": e.description,
                "properties": e.properties
            } for e in self.entities],
            "operations": [{
                "name": o.name,
                "operation_type": o.operation_type.value,
                "endpoint": o.endpoint,
                "method": o.method,
                "parameters": o.parameters,
                "returns": o.returns,
                "description": o.description
            } for o in self.operations],
            "constraints": [{
                "name": c.name,
                "constraint_type": c.constraint_type.value,
                "description": c.description,
                "properties": c.properties
            } for c in self.constraints],
            "language": self.language,
            "framework": self.framework,
            "notes": self.notes
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class RequirementParser:
    _counter_patterns = [
        r"(计数器|counter)",
        r"(增加|inc|increment)",
        r"(获取|get|read)",
        r"(并发|concurrency|thread)",
    ]
    
    _stack_patterns = [
        r"(栈|stack|堆栈)",
        r"(push|pop|入栈|出栈)",
        r"(top|peek)",
    ]
    
    _queue_patterns = [
        r"(队列|queue)",
        r"(enqueue|dequeue|入队|出队)",
        r"(FIFO|先进先出)",
    ]
    
    _constraint_patterns = [
        (r"(线性一致|linearizable)", ConstraintType.LINEARIZABILITY),
        (r"(一致性|consistency)", ConstraintType.CONSISTENCY),
        (r"(原子|atomic)", ConstraintType.ATOMICITY),
        (r"(锁|lock)", ConstraintType.LOCKING),
        (r"(线程安全|thread safe)", ConstraintType.THREAD_SAFETY),
        (r"(不丢失更新|no lost updates|不会丢失更新)", ConstraintType.NO_LOST_UPDATES),
        (r"(顺序|order)", ConstraintType.ORDER_PRESERVING),
        (r"(公平|fair)", ConstraintType.FAIRNESS),
    ]
    
    def __init__(self):
        self._templates = {
            "counter": self._parse_counter,
            "stack": self._parse_stack,
            "queue": self._parse_queue,
        }
    
    def parse(self, natural_language: str) -> ParsedRequirement:
        natural_language = natural_language.lower()
        
        template = self._detect_template(natural_language)
        if template:
            return self._templates[template](natural_language)
        
        return self._parse_generic(natural_language)
    
    def _detect_template(self, text: str) -> Optional[str]:
        if any(re.search(pattern, text) for pattern in self._counter_patterns):
            return "counter"
        if any(re.search(pattern, text) for pattern in self._stack_patterns):
            return "stack"
        if any(re.search(pattern, text) for pattern in self._queue_patterns):
            return "queue"
        return None
    
    def _parse_counter(self, text: str) -> ParsedRequirement:
        entities = [
            Entity(
                name="Counter",
                entity_type=EntityType.COUNTER,
                description="A thread-safe counter that supports increment, decrement, reset and get operations",
                properties={"initial_value": 0}
            )
        ]
        
        operations = [
            Operation(
                name="increment",
                operation_type=OperationType.INCREMENT,
                endpoint="/inc",
                method="POST",
                parameters=[],
                returns="int",
                description="Increment the counter by 1"
            ),
            Operation(
                name="decrement",
                operation_type=OperationType.DECREMENT,
                endpoint="/dec",
                method="POST",
                parameters=[],
                returns="int",
                description="Decrement the counter by 1"
            ),
            Operation(
                name="reset",
                operation_type=OperationType.RESET,
                endpoint="/reset",
                method="POST",
                parameters=[],
                returns="int",
                description="Reset the counter to 0"
            ),
            Operation(
                name="get",
                operation_type=OperationType.GET,
                endpoint="/get",
                method="GET",
                parameters=[],
                returns="int",
                description="Get the current counter value"
            )
        ]
        
        constraints = self._extract_constraints(text)
        if not constraints:
            constraints.append(
                Constraint(
                    name="NoLostUpdates",
                    constraint_type=ConstraintType.NO_LOST_UPDATES,
                    description="Concurrent increment operations should not lose updates",
                    properties={"requires_lock": True}
                )
            )
        
        return ParsedRequirement(
            name="CounterService",
            description="HTTP counter service with increment and get operations",
            entities=entities,
            operations=operations,
            constraints=constraints,
            notes=["Built-in counter template"]
        )
    
    def _parse_stack(self, text: str) -> ParsedRequirement:
        entities = [
            Entity(
                name="Stack",
                entity_type=EntityType.STACK,
                description="A thread-safe stack that supports push, pop, and peek operations",
                properties={"max_size": None}
            )
        ]
        
        operations = [
            Operation(
                name="push",
                operation_type=OperationType.PUSH,
                endpoint="/push",
                method="POST",
                parameters=["item"],
                returns="void",
                description="Push an item onto the stack"
            ),
            Operation(
                name="pop",
                operation_type=OperationType.POP,
                endpoint="/pop",
                method="POST",
                parameters=[],
                returns="any",
                description="Remove and return the top item from the stack"
            ),
            Operation(
                name="peek",
                operation_type=OperationType.READ,
                endpoint="/peek",
                method="GET",
                parameters=[],
                returns="any",
                description="Return the top item without removing it"
            )
        ]
        
        constraints = self._extract_constraints(text)
        if not constraints:
            constraints.append(
                Constraint(
                    name="OrderPreserving",
                    constraint_type=ConstraintType.ORDER_PRESERVING,
                    description="Last-in-first-out order is preserved",
                    properties={"order": "LIFO"}
                )
            )
        
        return ParsedRequirement(
            name="StackService",
            description="HTTP stack service with push, pop, and peek operations",
            entities=entities,
            operations=operations,
            constraints=constraints,
            notes=["Built-in stack template"]
        )
    
    def _parse_queue(self, text: str) -> ParsedRequirement:
        entities = [
            Entity(
                name="Queue",
                entity_type=EntityType.QUEUE,
                description="A thread-safe queue that supports enqueue, dequeue operations",
                properties={"max_size": None}
            )
        ]
        
        operations = [
            Operation(
                name="enqueue",
                operation_type=OperationType.ENQUEUE,
                endpoint="/enqueue",
                method="POST",
                parameters=["item"],
                returns="void",
                description="Add an item to the end of the queue"
            ),
            Operation(
                name="dequeue",
                operation_type=OperationType.DEQUEUE,
                endpoint="/dequeue",
                method="POST",
                parameters=[],
                returns="any",
                description="Remove and return the front item from the queue"
            )
        ]
        
        constraints = self._extract_constraints(text)
        if not constraints:
            constraints.append(
                Constraint(
                    name="OrderPreserving",
                    constraint_type=ConstraintType.ORDER_PRESERVING,
                    description="First-in-first-out order is preserved",
                    properties={"order": "FIFO"}
                )
            )
        
        return ParsedRequirement(
            name="QueueService",
            description="HTTP queue service with enqueue and dequeue operations",
            entities=entities,
            operations=operations,
            constraints=constraints,
            notes=["Built-in queue template"]
        )
    
    def _parse_generic(self, text: str) -> ParsedRequirement:
        entities = self._extract_entities(text)
        operations = self._extract_operations(text)
        constraints = self._extract_constraints(text)
        
        return ParsedRequirement(
            name="GenericService",
            description=text[:100] + "..." if len(text) > 100 else text,
            entities=entities,
            operations=operations,
            constraints=constraints,
            notes=["Generic parsing template"]
        )
    
    def _extract_entities(self, text: str) -> List[Entity]:
        entities = []
        
        service_pattern = r"(service|server|api)"
        if re.search(service_pattern, text):
            entities.append(Entity(
                name="Service",
                entity_type=EntityType.SERVICE,
                description="Main service entity"
            ))
        
        db_pattern = r"(database|db|storage)"
        if re.search(db_pattern, text):
            entities.append(Entity(
                name="Database",
                entity_type=EntityType.DATABASE,
                description="Data storage"
            ))
        
        return entities
    
    def _extract_operations(self, text: str) -> List[Operation]:
        operations = []
        
        if re.search(r"(create|add|new)", text):
            operations.append(Operation(
                name="create",
                operation_type=OperationType.CREATE,
                endpoint="/create",
                method="POST"
            ))
        
        if re.search(r"(read|get|fetch)", text):
            operations.append(Operation(
                name="read",
                operation_type=OperationType.READ,
                endpoint="/read",
                method="GET"
            ))
        
        if re.search(r"(update|modify|change)", text):
            operations.append(Operation(
                name="update",
                operation_type=OperationType.UPDATE,
                endpoint="/update",
                method="PUT"
            ))
        
        if re.search(r"(delete|remove|destroy)", text):
            operations.append(Operation(
                name="delete",
                operation_type=OperationType.DELETE,
                endpoint="/delete",
                method="DELETE"
            ))
        
        return operations
    
    def _extract_constraints(self, text: str) -> List[Constraint]:
        constraints = []
        
        for pattern, constraint_type in self._constraint_patterns:
            if re.search(pattern, text):
                constraints.append(Constraint(
                    name=constraint_type.value.replace("_", " ").title(),
                    constraint_type=constraint_type,
                    description=f"Constraint: {constraint_type.value.replace('_', ' ')}"
                ))
        
        return constraints
    
    def parse_counter_example(self) -> ParsedRequirement:
        return self.parse(
            "实现 HTTP 计数器服务，两个操作：POST /inc 增加计数；GET /get 返回当前值。"
            "要求并发调用不会丢失更新。"
        )
    
    def parse_stack_example(self) -> ParsedRequirement:
        return self.parse(
            "实现一个线程安全的栈服务，支持 push、pop、peek 操作。"
            "要求保持后进先出顺序。"
        )
    
    def parse_queue_example(self) -> ParsedRequirement:
        return self.parse(
            "实现一个消息队列服务，支持 enqueue 和 dequeue 操作。"
            "要求保持先进先出顺序，支持并发访问。"
        )