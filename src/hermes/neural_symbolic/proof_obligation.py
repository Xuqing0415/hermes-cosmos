"""
Proof Obligation Generator - Converts path conditions to SMT formulas
"""

import ast
from typing import List, Dict, Optional, Any
import structlog

from hermes.neural_symbolic.types import (
    PathCondition, 
    VerificationCondition, 
    ProofTarget
)

logger = structlog.get_logger()


class ProofObligationGenerator:
    """
    Generates verification conditions from code and path conditions.
    
    Converts:
    - Code AST to path conditions
    - Path conditions to SMT-LIB format verification conditions
    """
    
    def __init__(self):
        self._obligations = []
    
    def generate_from_code(self, code: str, function_name: str) -> List[VerificationCondition]:
        """
        Generate verification conditions from Python code.
        
        Args:
            code: Python source code
            function_name: Target function name
        
        Returns:
            List of verification conditions
        """
        try:
            tree = ast.parse(code)
            return self._analyze_ast(tree, function_name)
        except SyntaxError as e:
            logger.error("Failed to parse code", error=str(e))
            return []
    
    def _analyze_ast(self, tree: ast.AST, function_name: str) -> List[VerificationCondition]:
        """Analyze AST to extract verification conditions"""
        conditions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                conditions.extend(self._analyze_function(node))
        
        return conditions
    
    def _analyze_function(self, func: ast.FunctionDef) -> List[VerificationCondition]:
        """Analyze a function definition for verification conditions"""
        conditions = []
        
        for stmt in func.body:
            if isinstance(stmt, ast.Return):
                conditions.extend(self._analyze_return(stmt, func))
            elif isinstance(stmt, ast.If):
                conditions.extend(self._analyze_if(stmt, func))
            elif isinstance(stmt, ast.Assign):
                conditions.extend(self._analyze_assign(stmt, func))
        
        return conditions
    
    def _analyze_return(self, stmt: ast.Return, func: ast.FunctionDef) -> List[VerificationCondition]:
        """Analyze return statement for verification conditions"""
        conditions = []
        
        if isinstance(stmt.value, ast.BinOp):
            if isinstance(stmt.value.op, ast.Div):
                divisor = self._ast_to_smt(stmt.value.right)
                vc = VerificationCondition(
                    id=f"{func.name}_div_check",
                    smt_formula=f"(not (= {divisor} 0))",
                    description=f"Ensure divisor {divisor} is not zero in function {func.name}",
                    source=f"return {self._ast_to_python(stmt.value)}"
                )
                conditions.append(vc)
        
        return conditions
    
    def _analyze_if(self, stmt: ast.If, func: ast.FunctionDef) -> List[VerificationCondition]:
        """Analyze if statement for verification conditions"""
        conditions = []
        condition_str = self._ast_to_smt(stmt.test)
        
        vc = VerificationCondition(
            id=f"{func.name}_if_{hash(condition_str) % 1000}",
            smt_formula=condition_str,
            description=f"Path condition for if statement in {func.name}",
            source=f"if {self._ast_to_python(stmt.test)}:"
        )
        conditions.append(vc)
        
        return conditions
    
    def _analyze_assign(self, stmt: ast.Assign, func: ast.FunctionDef) -> List[VerificationCondition]:
        """Analyze assignment for verification conditions"""
        conditions = []
        return conditions
    
    def _ast_to_smt(self, node: ast.AST) -> str:
        """Convert AST node to SMT-LIB format"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_smt(node.left)
            right = self._ast_to_smt(node.right)
            op = self._binop_to_smt(node.op)
            return f"({op} {left} {right})"
        elif isinstance(node, ast.Compare):
            left = self._ast_to_smt(node.left)
            right = self._ast_to_smt(node.comparators[0])
            op = self._cmpop_to_smt(node.ops[0])
            return f"({op} {left} {right})"
        elif isinstance(node, ast.UnaryOp):
            operand = self._ast_to_smt(node.operand)
            op = self._unaryop_to_smt(node.op)
            return f"({op} {operand})"
        else:
            return str(node)
    
    def _ast_to_python(self, node: ast.AST) -> str:
        """Convert AST node back to Python code"""
        return ast.unparse(node)
    
    def _binop_to_smt(self, op: ast.operator) -> str:
        """Convert binary operator to SMT-LIB"""
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        elif isinstance(op, ast.Mod):
            return "mod"
        else:
            return str(op)
    
    def _cmpop_to_smt(self, op: ast.cmpop) -> str:
        """Convert comparison operator to SMT-LIB"""
        if isinstance(op, ast.Eq):
            return "="
        elif isinstance(op, ast.NotEq):
            return "distinct"
        elif isinstance(op, ast.Lt):
            return "<"
        elif isinstance(op, ast.LtE):
            return "<="
        elif isinstance(op, ast.Gt):
            return ">"
        elif isinstance(op, ast.GtE):
            return ">="
        else:
            return str(op)
    
    def _unaryop_to_smt(self, op: ast.unaryop) -> str:
        """Convert unary operator to SMT-LIB"""
        if isinstance(op, ast.Not):
            return "not"
        elif isinstance(op, ast.USub):
            return "-"
        else:
            return str(op)
    
    def generate_from_path_condition(
        self,
        path_condition: PathCondition,
        precondition: Optional[str] = None,
        postcondition: Optional[str] = None
    ) -> VerificationCondition:
        """
        Generate verification condition from path condition.
        
        Args:
            path_condition: Path condition from symbolic execution
            precondition: Optional precondition
            postcondition: Optional postcondition
        
        Returns:
            VerificationCondition
        """
        if precondition and postcondition:
            smt_formula = f"(=> (and {precondition} {path_condition.condition}) {postcondition})"
        elif postcondition:
            smt_formula = f"(=> {path_condition.condition} {postcondition})"
        else:
            smt_formula = path_condition.condition
        
        return VerificationCondition(
            id=f"vc_{hash(smt_formula) % 100000}",
            smt_formula=smt_formula,
            description=f"Verification condition for: {path_condition.condition}",
            source=str(path_condition)
        )
    
    def generate_proof_targets(
        self,
        code: str,
        function_name: str
    ) -> List[ProofTarget]:
        """
        Generate proof targets from code.
        
        Args:
            code: Python source code
            function_name: Target function name
        
        Returns:
            List of ProofTarget objects
        """
        vcs = self.generate_from_code(code, function_name)
        targets = []
        
        for vc in vcs:
            path_cond = PathCondition(vc.smt_formula)
            target = ProofTarget(
                id=vc.id,
                function_name=function_name,
                path_condition=path_cond,
                precondition=None,
                postcondition=None,
                risk_score=self._calculate_risk(vc),
                priority=int(self._calculate_risk(vc) * 10)
            )
            targets.append(target)
        
        return sorted(targets, key=lambda t: t.priority, reverse=True)
    
    def _calculate_risk(self, vc: VerificationCondition) -> float:
        """Calculate risk score for a verification condition"""
        score = 0.0
        
        if "div" in vc.description.lower() or "/" in vc.smt_formula:
            score += 0.8
        if "0" in vc.smt_formula:
            score += 0.5
        if "assert" in vc.description.lower():
            score += 0.3
        
        return min(score, 1.0)
