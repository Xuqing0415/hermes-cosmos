from typing import Dict, List, Optional, Any, Tuple, Set
import re
import time
import json

from .types import (
    MLIRDialect, IRNode, IRNodeType, MemRefType, SymbolicValue,
    VerificationResult, VerificationError, ErrorType, VerificationStatus
)

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class BasicBlock:
    def __init__(self, name: str):
        self.name = name
        self.instructions: List[Dict[str, Any]] = []
        self.predecessors: List[str] = []
        self.successors: List[str] = []
        self.terminator: Optional[Dict[str, Any]] = None
    
    def add_instruction(self, instr: Dict[str, Any]):
        self.instructions.append(instr)
    
    def set_terminator(self, terminator: Dict[str, Any]):
        self.terminator = terminator
    
    def __repr__(self):
        return f"BasicBlock(name={self.name}, instructions={len(self.instructions)})"


class MLIRSymbolicExecutor:
    def __init__(self):
        self.symbolic_memory: Dict[str, MemRefType] = {}
        self.symbolic_values: Dict[str, Any] = {}
        self.current_block: Optional[str] = None
        self.results: VerificationResult = VerificationResult(status="passed")
        self._z3_context = z3.Context() if Z3_AVAILABLE else None
        self._cfg: Dict[str, BasicBlock] = {}
        self._entry_block: Optional[str] = None
        self._visited_paths: Set[str] = set()
    
    def parse_mlir_to_cfg(self, mlir_code: str) -> Dict[str, BasicBlock]:
        self._cfg = {}
        self._entry_block = None
        
        lines = mlir_code.strip().split('\n')
        current_block = BasicBlock("entry")
        self._cfg["entry"] = current_block
        self._entry_block = "entry"
        
        block_count = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('func.func') or line.startswith('}'):
                continue
            
            if line.startswith('^'):
                match = re.match(r'\^(\w+)\s*:', line)
                if match:
                    self._finalize_block(current_block)
                    block_name = match.group(1)
                    current_block = BasicBlock(block_name)
                    self._cfg[block_name] = current_block
                    block_count += 1
                continue
            
            if 'cf.br' in line:
                terminator = self._parse_terminator(line)
                current_block.set_terminator(terminator)
                self._finalize_block(current_block)
                current_block = BasicBlock(f"unnamed_{block_count}")
                block_count += 1
            
            elif 'cf.cond_br' in line:
                terminator = self._parse_terminator(line)
                current_block.set_terminator(terminator)
                self._finalize_block(current_block)
                current_block = BasicBlock(f"unnamed_{block_count}")
                block_count += 1
            
            elif 'cf.return' in line or 'return' in line:
                terminator = {"type": "return", "line": line}
                current_block.set_terminator(terminator)
                self._finalize_block(current_block)
                current_block = BasicBlock(f"unnamed_{block_count}")
                block_count += 1
            
            else:
                instr = self._parse_instruction(line)
                if instr:
                    current_block.add_instruction(instr)
        
        self._build_cfg_edges()
        return self._cfg
    
    def _parse_instruction(self, line: str) -> Optional[Dict[str, Any]]:
        if 'arith.constant' in line:
            match = re.search(r'%(\w+)\s*=\s*arith\.constant\s*(-?\d+)\s*:\s*(\w+)', line)
            if match:
                return {
                    "type": "constant",
                    "result": match.group(1),
                    "value": int(match.group(2)),
                    "dtype": match.group(3)
                }
        
        elif 'arith.addi' in line:
            match = re.search(r'%(\w+)\s*=\s*arith\.addi\s*%(\w+)\s*,\s*%(\w+)\s*:\s*(\w+)', line)
            if match:
                return {
                    "type": "addi",
                    "result": match.group(1),
                    "lhs": match.group(2),
                    "rhs": match.group(3),
                    "dtype": match.group(4)
                }
        
        elif 'arith.subi' in line:
            match = re.search(r'%(\w+)\s*=\s*arith\.subi\s*%(\w+)\s*,\s*%(\w+)\s*:\s*(\w+)', line)
            if match:
                return {
                    "type": "subi",
                    "result": match.group(1),
                    "lhs": match.group(2),
                    "rhs": match.group(3),
                    "dtype": match.group(4)
                }
        
        elif 'arith.muli' in line:
            match = re.search(r'%(\w+)\s*=\s*arith\.muli\s*%(\w+)\s*,\s*%(\w+)\s*:\s*(\w+)', line)
            if match:
                return {
                    "type": "muli",
                    "result": match.group(1),
                    "lhs": match.group(2),
                    "rhs": match.group(3),
                    "dtype": match.group(4)
                }
        
        elif 'arith.cmpi' in line:
            match = re.search(r'%(\w+)\s*=\s*arith\.cmpi\s*(\w+)\s*,\s*%(\w+)\s*,\s*%(\w+)\s*:\s*(\w+)', line)
            if match:
                return {
                    "type": "cmpi",
                    "result": match.group(1),
                    "predicate": match.group(2),
                    "lhs": match.group(3),
                    "rhs": match.group(4),
                    "dtype": match.group(5)
                }
        
        elif 'memref.load' in line:
            match = re.search(r'%(\w+)\s*=\s*memref\.load\s*%(\w+)\[(\d+)\]\s*:\s*(.*)', line)
            if match:
                return {
                    "type": "load",
                    "result": match.group(1),
                    "memref": match.group(2),
                    "index": int(match.group(3)),
                    "memref_type": match.group(4)
                }
        
        elif 'memref.store' in line:
            match = re.search(r'memref\.store\s*%\w+\s*,\s*%(\w+)\[(\d+)\]\s*:\s*(.*)', line)
            if match:
                return {
                    "type": "store",
                    "memref": match.group(1),
                    "index": int(match.group(2)),
                    "memref_type": match.group(3)
                }
        
        return None
    
    def _parse_terminator(self, line: str) -> Dict[str, Any]:
        if 'cf.br' in line and 'cf.cond_br' not in line:
            match = re.search(r'cf\.br\s*\^(\w+)', line)
            if match:
                return {"type": "br", "target": match.group(1)}
        
        elif 'cf.cond_br' in line:
            match = re.search(r'cf\.cond_br\s*%(\w+)\s*,\s*\^(\w+)\s*,\s*\^(\w+)', line)
            if match:
                return {
                    "type": "cond_br",
                    "condition": match.group(1),
                    "true_target": match.group(2),
                    "false_target": match.group(3)
                }
        
        return {"type": "unknown"}
    
    def _finalize_block(self, block: BasicBlock):
        if block.name not in self._cfg:
            self._cfg[block.name] = block
    
    def _build_cfg_edges(self):
        for block_name, block in self._cfg.items():
            if block.terminator:
                if block.terminator["type"] == "br":
                    target = block.terminator["target"]
                    if target in self._cfg:
                        block.successors.append(target)
                        self._cfg[target].predecessors.append(block_name)
                
                elif block.terminator["type"] == "cond_br":
                    true_target = block.terminator["true_target"]
                    false_target = block.terminator["false_target"]
                    if true_target in self._cfg:
                        block.successors.append(true_target)
                        self._cfg[true_target].predecessors.append(block_name)
                    if false_target in self._cfg:
                        block.successors.append(false_target)
                        self._cfg[false_target].predecessors.append(block_name)
    
    def generate_cfg_dot(self, filename: str = "cfg.dot") -> str:
        dot_content = "digraph CFG {\n"
        dot_content += "    node [shape=box, style=filled, color=lightblue];\n"
        
        for block_name, block in self._cfg.items():
            label = f"{block_name}\\n"
            for instr in block.instructions[:3]:
                label += f"{instr['type']}\\n"
            if len(block.instructions) > 3:
                label += f"... ({len(block.instructions)})\\n"
            if block.terminator:
                label += f"→ {block.terminator.get('type', '')}"
            
            dot_content += f'    "{block_name}" [label="{label}"];\n'
            
            for successor in block.successors:
                dot_content += f'    "{block_name}" -> "{successor}";\n'
        
        dot_content += "}\n"
        
        with open(filename, 'w') as f:
            f.write(dot_content)
        
        return dot_content
    
    def _get_z3_var(self, name: str, dtype: str) -> Any:
        if not Z3_AVAILABLE:
            if name not in self.symbolic_values:
                self.symbolic_values[name] = f"SymbolicVar({name}, {dtype})"
            return self.symbolic_values[name]
        
        if name not in self.symbolic_values:
            bit_width = self._get_bit_width(dtype)
            if bit_width > 0:
                self.symbolic_values[name] = z3.BitVec(name, bit_width, ctx=self._z3_context)
            else:
                self.symbolic_values[name] = z3.Int(name, ctx=self._z3_context)
        return self.symbolic_values[name]
    
    def _get_bit_width(self, dtype: str) -> int:
        if dtype.startswith('i'):
            try:
                return int(dtype[1:])
            except ValueError:
                return 0
        return 0
    
    def _is_signed(self, dtype: str) -> bool:
        return dtype.startswith('i')
    
    def execute(self, mlir_code: str) -> VerificationResult:
        start_time = time.time()
        
        self.symbolic_memory = {}
        self.symbolic_values = {}
        self.results = VerificationResult(status=VerificationStatus.PASSED)
        self._visited_paths = set()
        
        self.parse_mlir_to_cfg(mlir_code)
        self._parse_function_params_from_code(mlir_code)
        
        if self._entry_block:
            path_condition = z3.BoolVal(True, ctx=self._z3_context) if Z3_AVAILABLE else None
            self._symbolic_execute_block(self._entry_block, path_condition)
        
        self.results.execution_time = time.time() - start_time
        return self.results
    
    def _parse_function_params_from_code(self, mlir_code: str):
        match = re.search(r'func\.func @(\w+)\((.*?)\)', mlir_code)
        if match:
            params_str = match.group(2)
            params = params_str.split(',')
            for param in params:
                param = param.strip()
                if param:
                    parts = param.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip().replace('%', '')
                        type_str = parts[1].strip()
                        
                        if type_str.startswith('memref'):
                            memref_type = self._parse_memref_type(type_str)
                            self.symbolic_memory[name] = memref_type
                        elif type_str.startswith('i'):
                            self._get_z3_var(name, type_str)
                        elif type_str.startswith('f'):
                            self._get_z3_var(name, type_str)
    
    def _parse_memref_type(self, type_str: str) -> MemRefType:
        match = re.match(r'memref<(\[.*?\])x(\w+)>', type_str)
        if match:
            shape_str = match.group(1)
            element_type = match.group(2)
            
            shape = []
            dims = shape_str.strip('[]').split('x')
            for dim in dims:
                dim = dim.strip()
                if dim.isdigit():
                    shape.append(int(dim))
                else:
                    shape.append(-1)
            
            return MemRefType(shape=shape, element_type=element_type)
        
        return MemRefType(shape=[], element_type="f32")
    
    def _symbolic_execute_block(self, block_name: str, path_condition: Optional[z3.ExprRef]):
        if block_name not in self._cfg:
            return
        
        path_key = f"{block_name}:{hash(str(path_condition))}" if path_condition else f"{block_name}:none"
        if path_key in self._visited_paths:
            return
        self._visited_paths.add(path_key)
        
        block = self._cfg[block_name]
        
        for instr in block.instructions:
            self._execute_instruction(instr, path_condition)
        
        if block.terminator:
            self._execute_terminator(block.terminator, path_condition)
    
    def _execute_instruction(self, instr: Dict[str, Any], path_condition: Optional[z3.ExprRef]):
        instr_type = instr["type"]
        
        if instr_type == "load":
            self._check_memref_bounds(instr["memref"], instr["index"], path_condition)
            return
        
        elif instr_type == "store":
            self._check_memref_bounds(instr["memref"], instr["index"], path_condition)
            return
        
        if not Z3_AVAILABLE:
            if instr_type == "constant":
                self.symbolic_values[instr["result"]] = f"Constant({instr['value']}, {instr['dtype']})"
            elif instr_type in ["addi", "subi", "muli", "cmpi"]:
                lhs = self._get_z3_var(instr["lhs"], instr["dtype"])
                rhs = self._get_z3_var(instr["rhs"], instr["dtype"])
                self.symbolic_values[instr["result"]] = f"{instr_type}({lhs}, {rhs})"
            return
        
        if instr_type == "constant":
            bit_width = self._get_bit_width(instr["dtype"])
            if bit_width > 0:
                self.symbolic_values[instr["result"]] = z3.BitVecVal(instr["value"], bit_width, ctx=self._z3_context)
            else:
                self.symbolic_values[instr["result"]] = z3.IntVal(instr["value"], ctx=self._z3_context)
        
        elif instr_type == "addi":
            lhs = self._get_z3_var(instr["lhs"], instr["dtype"])
            rhs = self._get_z3_var(instr["rhs"], instr["dtype"])
            self.symbolic_values[instr["result"]] = lhs + rhs
            self._check_addi_overflow(instr, lhs, rhs)
        
        elif instr_type == "subi":
            lhs = self._get_z3_var(instr["lhs"], instr["dtype"])
            rhs = self._get_z3_var(instr["rhs"], instr["dtype"])
            self.symbolic_values[instr["result"]] = lhs - rhs
        
        elif instr_type == "muli":
            lhs = self._get_z3_var(instr["lhs"], instr["dtype"])
            rhs = self._get_z3_var(instr["rhs"], instr["dtype"])
            self.symbolic_values[instr["result"]] = lhs * rhs
            self._check_muli_overflow(instr, lhs, rhs)
        
        elif instr_type == "cmpi":
            lhs = self._get_z3_var(instr["lhs"], instr["dtype"])
            rhs = self._get_z3_var(instr["rhs"], instr["dtype"])
            predicate = instr["predicate"]
            
            if predicate == "eq":
                self.symbolic_values[instr["result"]] = lhs == rhs
            elif predicate == "ne":
                self.symbolic_values[instr["result"]] = lhs != rhs
            elif predicate == "slt":
                self.symbolic_values[instr["result"]] = z3.SLT(lhs, rhs)
            elif predicate == "sle":
                self.symbolic_values[instr["result"]] = z3.SLE(lhs, rhs)
            elif predicate == "sgt":
                self.symbolic_values[instr["result"]] = z3.SGT(lhs, rhs)
            elif predicate == "sge":
                self.symbolic_values[instr["result"]] = z3.SGE(lhs, rhs)
    
    def _check_addi_overflow(self, instr: Dict[str, Any], lhs: z3.ExprRef, rhs: z3.ExprRef):
        dtype = instr["dtype"]
        bit_width = self._get_bit_width(dtype)
        if bit_width == 0:
            return
        
        solver = z3.Solver(ctx=self._z3_context)
        
        if self._is_signed(dtype):
            min_val = - (1 << (bit_width - 1))
            max_val = (1 << (bit_width - 1)) - 1
            solver.add(lhs >= min_val, lhs <= max_val, rhs >= min_val, rhs <= max_val)
            
            overflow_condition = z3.Or(
                z3.And(lhs >= 0, rhs >= 0, lhs + rhs < lhs),
                z3.And(lhs < 0, rhs < 0, lhs + rhs >= 0)
            )
        else:
            max_val = (1 << bit_width) - 1
            solver.add(lhs >= 0, lhs <= max_val, rhs >= 0, rhs <= max_val)
            overflow_condition = lhs + rhs < lhs
        
        solver.add(overflow_condition)
        
        if solver.check() == z3.sat:
            model = solver.model()
            self.results.add_error(
                error_type=ErrorType.UNDEFINED_BEHAVIOR,
                message=f"Integer overflow detected in arith.addi: %{instr['result']} = %{instr['lhs']} + %{instr['rhs']}",
                smt_query=str(solver),
                location=self._format_model(model)
            )
    
    def _check_muli_overflow(self, instr: Dict[str, Any], lhs: z3.ExprRef, rhs: z3.ExprRef):
        dtype = instr["dtype"]
        bit_width = self._get_bit_width(dtype)
        if bit_width == 0:
            return
        
        solver = z3.Solver(ctx=self._z3_context)
        
        if self._is_signed(dtype):
            min_val = - (1 << (bit_width - 1))
            max_val = (1 << (bit_width - 1)) - 1
            solver.add(lhs >= min_val, lhs <= max_val, rhs >= min_val, rhs <= max_val)
            
            overflow_condition = z3.Not(z3.And(
                z3.Implies(z3.And(lhs == 0, rhs == 0), lhs * rhs == 0),
                z3.Implies(z3.And(lhs > 0, rhs > 0), lhs * rhs > 0),
                z3.Implies(z3.And(lhs > 0, rhs < 0), lhs * rhs <= 0),
                z3.Implies(z3.And(lhs < 0, rhs > 0), lhs * rhs <= 0),
                z3.Implies(z3.And(lhs < 0, rhs < 0), lhs * rhs > 0)
            ))
        else:
            max_val = (1 << bit_width) - 1
            solver.add(lhs >= 0, lhs <= max_val, rhs >= 0, rhs <= max_val)
            overflow_condition = lhs * rhs < z3.If(lhs == 0, 0, z3.If(rhs == 0, 0, z3.And(lhs > 0, rhs > 0)))
        
        solver.add(overflow_condition)
        
        if solver.check() == z3.sat:
            model = solver.model()
            self.results.add_error(
                error_type=ErrorType.UNDEFINED_BEHAVIOR,
                message=f"Integer overflow detected in arith.muli: %{instr['result']} = %{instr['lhs']} * %{instr['rhs']}",
                smt_query=str(solver),
                location=self._format_model(model)
            )
    
    def _check_memref_bounds(self, memref_name: str, index: int, path_condition: Optional[z3.ExprRef]):
        if memref_name not in self.symbolic_memory:
            return
        
        memref_type = self.symbolic_memory[memref_name]
        if not memref_type.shape:
            return
        
        if memref_type.shape[0] > 0 and (index < 0 or index >= memref_type.shape[0]):
            self.results.add_error(
                error_type=ErrorType.INDEX_OUT_OF_BOUNDS,
                message=f"memref access index {index} out of bounds for memref {memref_name} with shape {memref_type.shape}"
            )
            return
        
        if Z3_AVAILABLE:
            solver = z3.Solver(ctx=self._z3_context)
            
            if path_condition is not None:
                solver.add(path_condition)
            
            for i, dim in enumerate(memref_type.shape):
                if dim > 0:
                    idx_var = z3.Int(f"idx_{memref_name}_{i}", ctx=self._z3_context)
                    solver.add(idx_var >= 0)
                    solver.add(idx_var < dim)
            
            solver.add(z3.IntVal(index, ctx=self._z3_context) >= 0)
            solver.add(z3.IntVal(index, ctx=self._z3_context) < memref_type.shape[0])
            
            if solver.check() == z3.unsat:
                self.results.add_error(
                    error_type=ErrorType.INDEX_OUT_OF_BOUNDS,
                    message=f"memref access index {index} out of bounds for memref {memref_name} with shape {memref_type.shape}",
                    smt_query=str(solver)
                )
    
    def _execute_terminator(self, terminator: Dict[str, Any], path_condition: Optional[z3.ExprRef]):
        if terminator["type"] == "br":
            self._symbolic_execute_block(terminator["target"], path_condition)
        
        elif terminator["type"] == "cond_br":
            if Z3_AVAILABLE:
                condition_var = self.symbolic_values.get(terminator["condition"])
                
                if condition_var is not None:
                    true_condition = z3.And(path_condition, condition_var) if path_condition else condition_var
                    false_condition = z3.And(path_condition, z3.Not(condition_var)) if path_condition else z3.Not(condition_var)
                    
                    self._symbolic_execute_block(terminator["true_target"], true_condition)
                    self._symbolic_execute_block(terminator["false_target"], false_condition)
                else:
                    self._symbolic_execute_block(terminator["true_target"], path_condition)
                    self._symbolic_execute_block(terminator["false_target"], path_condition)
    
    def _format_model(self, model: z3.ModelRef) -> str:
        assignments = []
        for decl in model:
            assignments.append(f"{decl.name()} = {model[decl]}")
        return ", ".join(assignments)
    
    def get_symbolic_state(self) -> Dict[str, Any]:
        return {
            "memory": {name: memref.shape for name, memref in self.symbolic_memory.items()},
            "values": {name: str(val) for name, val in self.symbolic_values.items()},
            "cfg_blocks": list(self._cfg.keys()),
            "visited_paths": len(self._visited_paths)
        }
    
    def get_cfg_info(self) -> Dict[str, Any]:
        info = {}
        for name, block in self._cfg.items():
            info[name] = {
                "instructions": len(block.instructions),
                "predecessors": block.predecessors,
                "successors": block.successors,
                "terminator": block.terminator["type"] if block.terminator else None
            }
        return info