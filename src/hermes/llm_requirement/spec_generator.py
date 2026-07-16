from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import jinja2
import os

from .requirement_parser import ParsedRequirement, Entity, Operation, Constraint


@dataclass
class TLAPlusSpec:
    name: str
    module_content: str
    properties: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    temporal_properties: List[str] = field(default_factory=list)
    
    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.module_content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "properties": self.properties,
            "invariants": self.invariants,
            "temporal_properties": self.temporal_properties
        }


class SpecGenerator:
    _counter_tla_template = """---- MODULE {{ spec_name }} ----

CONSTANTS
    MaxValue

VARIABLES
    value

Init == value = 0

Increment ==
    /\\ value < MaxValue
    /\\ value' = value + 1

Get ==
    /\\ value' = value

Next ==
    Increment \\/ Get

TypeOK ==
    /\\ value \\in 0..MaxValue

NoLostUpdates ==
    \\A x, y \\in 0..MaxValue:
        /\\ x < y
        => (value = x ~> value = y)
            => (\\E k \\in x..y-1: value = k ~> value = k+1)

Spec ==
    Init /\\ [][Next]_<<value>>

THEOREM Spec => []TypeOK
THEOREM Spec => []NoLostUpdates

====
"""
    
    _stack_tla_template = """---- MODULE {{ spec_name }} ----

CONSTANTS
    MaxSize
    Items

VARIABLES
    stack

Init == stack = <<>>

Push(item) ==
    /\\ Len(stack) < MaxSize
    /\\ stack' = Append(stack, item)

Pop ==
    /\\ Len(stack) > 0
    /\\ stack' = SubSeq(stack, 1, Len(stack) - 1)

Peek ==
    /\\ Len(stack) > 0
    /\\ stack' = stack

Next ==
    \\E item \\in Items: Push(item)
    \\/ Pop
    \\/ Peek

TypeOK ==
    /\\ stack \\in [0..MaxSize -> Items]

LIFO ==
    \\A item1, item2 \\in Items:
        (\\E i, j: i < j /\\ stack[i] = item1 /\\ stack[j] = item2)
        => (Pop => (stack'[Len(stack')] # item1) \\/ (stack'[Len(stack')] = item1 /\\ stack'[Len(stack')-1] = item2))

Spec ==
    Init /\\ [][Next]_<<stack>>

THEOREM Spec => []TypeOK

====
"""
    
    _queue_tla_template = """---- MODULE {{ spec_name }} ----

CONSTANTS
    MaxSize
    Items

VARIABLES
    queue
    head
    tail

Init ==
    /\\ queue = [0..MaxSize-1 -> Items]
    /\\ head = 0
    /\\ tail = 0

Enqueue(item) ==
    /\\ (tail + 1) % MaxSize # head
    /\\ queue' = [queue EXCEPT ![tail] = item]
    /\\ tail' = (tail + 1) % MaxSize
    /\\ head' = head

Dequeue ==
    /\\ head # tail
    /\\ queue' = queue
    /\\ head' = (head + 1) % MaxSize
    /\\ tail' = tail

Next ==
    \\E item \\in Items: Enqueue(item)
    \\/ Dequeue

TypeOK ==
    /\\ queue \\in [0..MaxSize-1 -> Items]
    /\\ head \\in 0..MaxSize-1
    /\\ tail \\in 0..MaxSize-1

FIFO ==
    TRUE

Spec ==
    Init /\\ [][Next]_<<queue, head, tail>>

THEOREM Spec => []TypeOK

====
"""
    
    def __init__(self):
        self._templates = {
            "CounterService": self._generate_counter_spec,
            "StackService": self._generate_stack_spec,
            "QueueService": self._generate_queue_spec,
        }
    
    def generate(self, requirement: ParsedRequirement) -> TLAPlusSpec:
        template = self._templates.get(requirement.name)
        if template:
            return template(requirement)
        
        return self._generate_generic_spec(requirement)
    
    def _generate_counter_spec(self, requirement: ParsedRequirement) -> TLAPlusSpec:
        template = jinja2.Template(self._counter_tla_template)
        content = template.render(spec_name=requirement.name)
        
        return TLAPlusSpec(
            name=requirement.name,
            module_content=content,
            properties=["Increment", "Get"],
            invariants=["TypeOK", "NoLostUpdates"],
            temporal_properties=["Spec => []TypeOK", "Spec => []NoLostUpdates"]
        )
    
    def _generate_stack_spec(self, requirement: ParsedRequirement) -> TLAPlusSpec:
        template = jinja2.Template(self._stack_tla_template)
        content = template.render(spec_name=requirement.name)
        
        return TLAPlusSpec(
            name=requirement.name,
            module_content=content,
            properties=["Push", "Pop", "Peek"],
            invariants=["TypeOK", "LIFO"],
            temporal_properties=["Spec => []TypeOK"]
        )
    
    def _generate_queue_spec(self, requirement: ParsedRequirement) -> TLAPlusSpec:
        template = jinja2.Template(self._queue_tla_template)
        content = template.render(spec_name=requirement.name)
        
        return TLAPlusSpec(
            name=requirement.name,
            module_content=content,
            properties=["Enqueue", "Dequeue"],
            invariants=["TypeOK", "FIFO"],
            temporal_properties=["Spec => []TypeOK"]
        )
    
    def _generate_generic_spec(self, requirement: ParsedRequirement) -> TLAPlusSpec:
        operations = ", ".join(op.name for op in requirement.operations)
        invariants = ", ".join(c.name for c in requirement.constraints)
        
        content = f"""---- MODULE {requirement.name} ----

DESCRIPTION
    Generated from natural language requirement: {requirement.description[:50]}...

CONSTANTS
    None

VARIABLES
    state

Init == state = "initial"

Operations ==
    {operations}

Next ==
    TRUE

Spec ==
    Init /\\ [][Next]_<<state>>

====
"""
        
        return TLAPlusSpec(
            name=requirement.name,
            module_content=content,
            properties=[op.name for op in requirement.operations],
            invariants=[c.name for c in requirement.constraints],
            temporal_properties=[]
        )
    
    def generate_counter_tlaplus(self) -> TLAPlusSpec:
        from .requirement_parser import RequirementParser
        parser = RequirementParser()
        requirement = parser.parse_counter_example()
        return self.generate(requirement)
    
    def generate_stack_tlaplus(self) -> TLAPlusSpec:
        from .requirement_parser import RequirementParser
        parser = RequirementParser()
        requirement = parser.parse_stack_example()
        return self.generate(requirement)
    
    def generate_queue_tlaplus(self) -> TLAPlusSpec:
        from .requirement_parser import RequirementParser
        parser = RequirementParser()
        requirement = parser.parse_queue_example()
        return self.generate(requirement)