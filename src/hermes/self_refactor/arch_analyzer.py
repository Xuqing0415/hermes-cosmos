import ast
import os
import sys
from typing import List, Dict, Set, Tuple, Optional
import hashlib
import statistics

from .types import ArchMetrics, ModuleMetrics


class ArchAnalyzer:
    def __init__(self, source_dir: str = "src/hermes"):
        self.source_dir = source_dir
        self.module_metrics_cache: Dict[str, ModuleMetrics] = {}
        self.file_metrics_cache: Dict[str, ArchMetrics] = {}

    def analyze_all(self) -> List[ModuleMetrics]:
        modules = self._find_modules()
        all_metrics = []
        
        for module_path in modules:
            metrics = self.analyze_module(module_path)
            all_metrics.append(metrics)
        
        return all_metrics

    def analyze_module(self, module_path: str) -> ModuleMetrics:
        if module_path in self.module_metrics_cache:
            return self.module_metrics_cache[module_path]
        
        file_paths = self._find_python_files(module_path)
        file_metrics = []
        
        for fp in file_paths:
            fm = self.analyze_file(fp)
            file_metrics.append(fm)
        
        total_loc = sum(f.loc for f in file_metrics)
        total_functions = sum(f.function_count for f in file_metrics)
        total_classes = sum(f.class_count for f in file_metrics)
        avg_complexity = statistics.mean([f.cyclomatic_complexity for f in file_metrics]) if file_metrics else 0
        
        coupling = self._compute_coupling(module_path)
        cohesion = self._compute_cohesion(file_metrics)
        
        module_name = self._get_module_name(module_path)
        metrics = ModuleMetrics(
            module_name=module_name,
            file_metrics=file_metrics,
            total_loc=total_loc,
            total_functions=total_functions,
            total_classes=total_classes,
            avg_complexity=avg_complexity,
            coupling_score=coupling,
            cohesion_score=cohesion
        )
        
        self.module_metrics_cache[module_path] = metrics
        return metrics

    def analyze_file(self, file_path: str) -> ArchMetrics:
        if file_path in self.file_metrics_cache:
            return self.file_metrics_cache[file_path]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        
        loc = len(lines)
        complexity = self._compute_cyclomatic_complexity(tree)
        halstead = self._compute_halstead_volume(tree)
        mi = self._compute_maintainability_index(loc, complexity, halstead)
        func_count = self._count_functions(tree)
        class_count = self._count_classes(tree)
        
        metrics = ArchMetrics(
            file_path=file_path,
            loc=loc,
            cyclomatic_complexity=complexity,
            halstead_volume=halstead,
            maintainability_index=mi,
            function_count=func_count,
            class_count=class_count
        )
        
        self.file_metrics_cache[file_path] = metrics
        return metrics

    def _compute_cyclomatic_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
        
        return complexity

    def _compute_halstead_volume(self, tree: ast.AST) -> float:
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, 
                                 ast.Pow, ast.LShift, ast.RShift, ast.BitAnd,
                                 ast.BitOr, ast.BitXor, ast.FloorDiv, ast.MatMult,
                                 ast.And, ast.Or, ast.Not, ast.Invert,
                                 ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE,
                                 ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                operators.add(type(node).__name__)
                operator_count += 1
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))
                operand_count += 1
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
        
        if not operators:
            return 0.0
        
        n1 = len(operators)
        n2 = len(operands)
        N1 = operator_count
        N2 = operand_count
        
        vocabulary = n1 + n2
        volume = (N1 + N2) * (vocabulary ** 0.5) if vocabulary > 0 else 0
        return volume

    def _compute_maintainability_index(self, loc: int, complexity: int, halstead: float) -> float:
        if loc == 0:
            return 100.0
        
        mi = 171 - 5.2 * (halstead ** 0.5) - 0.23 * complexity - 16.2 * (loc ** 0.5)
        return max(0.0, min(100.0, mi))

    def _count_functions(self, tree: ast.AST) -> int:
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count += 1
        return count

    def _count_classes(self, tree: ast.AST) -> int:
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                count += 1
        return count

    def _compute_coupling(self, module_path: str) -> float:
        file_paths = self._find_python_files(module_path)
        all_imports = []
        
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        all_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        all_imports.append(node.module)
        
        external_imports = [imp for imp in all_imports if not imp.startswith('hermes')]
        total_imports = len(all_imports)
        
        if total_imports == 0:
            return 0.0
        
        return len(external_imports) / total_imports

    def _compute_cohesion(self, file_metrics: List[ArchMetrics]) -> float:
        if len(file_metrics) < 2:
            return 1.0
        
        total_functions = sum(f.function_count for f in file_metrics)
        if total_functions == 0:
            return 1.0
        
        avg_func_per_file = total_functions / len(file_metrics)
        ideal_func_per_file = total_functions ** 0.5
        
        return min(1.0, avg_func_per_file / ideal_func_per_file)

    def _find_modules(self) -> List[str]:
        modules = []
        for item in os.listdir(self.source_dir):
            item_path = os.path.join(self.source_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and not item.startswith('_'):
                modules.append(item_path)
        return modules

    def _find_python_files(self, dir_path: str) -> List[str]:
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith('.py') and not filename.startswith('_'):
                    files.append(os.path.join(root, filename))
        return files

    def _get_module_name(self, path: str) -> str:
        rel_path = os.path.relpath(path, self.source_dir)
        return rel_path.replace(os.sep, '.')

    def get_function_signature(self, file_path: str, function_name: str) -> Optional[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                func_source = ast.get_source_segment(content, node)
                if func_source:
                    return hashlib.sha256(func_source.encode()).hexdigest()[:16]
        
        return None

    def get_function_source(self, file_path: str, function_name: str) -> Optional[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return ast.get_source_segment(content, node)
        
        return None

    def get_class_source(self, file_path: str, class_name: str) -> Optional[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return ast.get_source_segment(content, node)
        
        return None