import ast
import os
import sys
from typing import List, Dict, Set, Tuple, Optional
import hashlib
import statistics

from .types import (
    SmellInstance,
    SmellType,
    SmellSeverity,
    CodeLocation,
    ArchMetrics,
    ModuleMetrics
)
from .arch_analyzer import ArchAnalyzer


class SmellDetector:
    LONG_FUNCTION_LINE_THRESHOLD = 50
    LONG_FUNCTION_COMPLEXITY_THRESHOLD = 10
    GOD_MODULE_FUNCTION_THRESHOLD = 20
    GOD_MODULE_CLASS_THRESHOLD = 10
    DUPLICATE_MIN_OCCURRENCES = 3
    DUPLICATE_MIN_LINES = 5
    DATA_CLUMP_MIN_OCCURRENCES = 3
    DATA_CLUMP_MIN_PARAMS = 3

    def __init__(self, analyzer: Optional[ArchAnalyzer] = None):
        self.analyzer = analyzer or ArchAnalyzer()
        self.detected_smells: List[SmellInstance] = []

    def detect_all(self) -> List[SmellInstance]:
        self.detected_smells = []
        
        self._detect_cycle_dependencies()
        self._detect_duplicate_code()
        self._detect_god_modules()
        self._detect_long_functions()
        self._detect_data_clumps()
        self._detect_shotgun_surgery()
        self._detect_feature_envy()
        
        self._prioritize_smells()
        return sorted(self.detected_smells, key=lambda s: s.priority, reverse=True)

    def _detect_cycle_dependencies(self):
        modules = self.analyzer._find_modules()
        import_graph: Dict[str, Set[str]] = {}
        
        for module_path in modules:
            module_name = self.analyzer._get_module_name(module_path)
            import_graph[module_name] = set()
            
            file_paths = self.analyzer._find_python_files(module_path)
            for fp in file_paths:
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        parts = node.module.split('.')
                        if len(parts) >= 2 and parts[0] == 'hermes':
                            imported_module = '.'.join(parts[:2])
                            if imported_module != module_name:
                                import_graph[module_name].add(imported_module)
        
        cycles = self._find_cycles(import_graph)
        
        for cycle in cycles:
            severity = SmellSeverity.CRITICAL if len(cycle) <= 2 else SmellSeverity.HIGH
            locations = []
            for mod in cycle:
                mod_path = os.path.join(self.analyzer.source_dir, mod.replace('.', os.sep))
                if os.path.exists(mod_path):
                    locations.append(CodeLocation(
                        file_path=mod_path,
                        line_start=1,
                        line_end=1
                    ))
            
            self.detected_smells.append(SmellInstance(
                smell_type=SmellType.CYCLE_DEPENDENCY,
                severity=severity,
                description=f": {' → '.join(cycle)}",
                locations=locations,
                metadata={'cycle': cycle},
                priority=100.0
            ))

    def _detect_duplicate_code(self):
        all_files = []
        for module_path in self.analyzer._find_modules():
            all_files.extend(self.analyzer._find_python_files(module_path))
        
        file_snippets: Dict[str, List[Tuple[str, int, int]]] = {}
        
        for fp in all_files:
            with open(fp, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i in range(len(lines) - self.DUPLICATE_MIN_LINES + 1):
                snippet = ''.join(lines[i:i+self.DUPLICATE_MIN_LINES])
                normalized = self._normalize_snippet(snippet)
                if len(normalized.strip()) < 20:
                    continue
                
                snippet_hash = hashlib.sha256(normalized.encode()).hexdigest()
                if snippet_hash not in file_snippets:
                    file_snippets[snippet_hash] = []
                file_snippets[snippet_hash].append((fp, i + 1, i + self.DUPLICATE_MIN_LINES))
        
        for snippet_hash, occurrences in file_snippets.items():
            if len(occurrences) >= self.DUPLICATE_MIN_OCCURRENCES:
                locations = [
                    CodeLocation(file_path=fp, line_start=start, line_end=end)
                    for fp, start, end in occurrences
                ]
                
                severity = SmellSeverity.HIGH if len(occurrences) >= 5 else SmellSeverity.MEDIUM
                self.detected_smells.append(SmellInstance(
                    smell_type=SmellType.DUPLICATE_CODE,
                    severity=severity,
                    description=f" {len(occurrences)} ",
                    locations=locations,
                    metadata={'occurrences': len(occurrences)},
                    priority=80.0 + len(occurrences) * 2
                ))

    def _detect_god_modules(self):
        module_metrics = self.analyzer.analyze_all()
        
        for metrics in module_metrics:
            is_god_function = metrics.total_functions > self.GOD_MODULE_FUNCTION_THRESHOLD
            is_god_class = metrics.total_classes > self.GOD_MODULE_CLASS_THRESHOLD
            
            if is_god_function or is_god_class:
                locations = []
                for fm in metrics.file_metrics:
                    locations.append(CodeLocation(
                        file_path=fm.file_path,
                        line_start=1,
                        line_end=fm.loc
                    ))
                
                desc_parts = []
                if is_god_function:
                    desc_parts.append(f" ({metrics.total_functions})")
                if is_god_class:
                    desc_parts.append(f" ({metrics.total_classes})")
                
                severity = SmellSeverity.HIGH if (is_god_function and is_god_class) else SmellSeverity.MEDIUM
                self.detected_smells.append(SmellInstance(
                    smell_type=SmellType.GOD_MODULE,
                    severity=severity,
                    description=f": {', '.join(desc_parts)}",
                    locations=locations,
                    metadata={
                        'total_functions': metrics.total_functions,
                        'total_classes': metrics.total_classes
                    },
                    priority=70.0 + metrics.total_functions
                ))

    def _detect_long_functions(self):
        module_metrics = self.analyzer.analyze_all()
        
        for metrics in module_metrics:
            for fm in metrics.file_metrics:
                with open(fm.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_source = ast.get_source_segment(content, node)
                        if func_source:
                            func_lines = len(func_source.split('\n'))
                            func_complexity = self._compute_function_complexity(node)
                            
                            if (func_lines > self.LONG_FUNCTION_LINE_THRESHOLD or 
                                func_complexity > self.LONG_FUNCTION_COMPLEXITY_THRESHOLD):
                                
                                line_start = node.lineno
                                line_end = node.lineno + func_lines
                                
                                severity = SmellSeverity.HIGH if (
                                    func_lines > self.LONG_FUNCTION_LINE_THRESHOLD * 2 or
                                    func_complexity > self.LONG_FUNCTION_COMPLEXITY_THRESHOLD * 2
                                ) else SmellSeverity.MEDIUM
                                
                                self.detected_smells.append(SmellInstance(
                                    smell_type=SmellType.LONG_FUNCTION,
                                    severity=severity,
                                    description=f" '{node.name}': {func_lines} ,  {func_complexity}",
                                    locations=[CodeLocation(
                                        file_path=fm.file_path,
                                        line_start=line_start,
                                        line_end=line_end
                                    )],
                                    metadata={
                                        'function_name': node.name,
                                        'line_count': func_lines,
                                        'complexity': func_complexity
                                    },
                                    priority=60.0 + func_complexity * 2
                                ))

    def _detect_data_clumps(self):
        all_files = []
        for module_path in self.analyzer._find_modules():
            all_files.extend(self.analyzer._find_python_files(module_path))
        
        param_patterns: Dict[str, List[Tuple[str, str]]] = {}
        
        for fp in all_files:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [arg.arg for arg in node.args.args]
                    if len(params) >= self.DATA_CLUMP_MIN_PARAMS:
                        param_key = ','.join(sorted(params))
                        if param_key not in param_patterns:
                            param_patterns[param_key] = []
                        param_patterns[param_key].append((fp, node.name))
        
        for param_key, occurrences in param_patterns.items():
            if len(occurrences) >= self.DATA_CLUMP_MIN_OCCURRENCES:
                locations = []
                for fp, func_name in occurrences:
                    with open(fp, 'r', encoding='utf-8') as f:
                        content = f.read()
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                            locations.append(CodeLocation(
                                file_path=fp,
                                line_start=node.lineno,
                                line_end=node.lineno + 5
                            ))
                            break
                
                severity = SmellSeverity.MEDIUM
                self.detected_smells.append(SmellInstance(
                    smell_type=SmellType.DATA_CLUMP,
                    severity=severity,
                    description=f":  '{param_key}'  {len(occurrences)} ",
                    locations=locations,
                    metadata={
                        'params': param_key.split(','),
                        'occurrences': len(occurrences)
                    },
                    priority=50.0 + len(occurrences) * 3
                ))

    def _detect_shotgun_surgery(self):
        all_files = []
        for module_path in self.analyzer._find_modules():
            all_files.extend(self.analyzer._find_python_files(module_path))
        
        constant_patterns: Dict[str, List[str]] = {}
        
        for fp in all_files:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            if isinstance(node.value, ast.Constant):
                                value_str = str(node.value.value)
                                pattern_key = f"{target.id}={value_str}"
                                if pattern_key not in constant_patterns:
                                    constant_patterns[pattern_key] = []
                                constant_patterns[pattern_key].append(fp)
        
        for pattern_key, files in constant_patterns.items():
            if len(files) >= 3:
                locations = []
                for fp in files:
                    with open(fp, 'r', encoding='utf-8') as f:
                        content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern_key in line:
                            locations.append(CodeLocation(
                                file_path=fp,
                                line_start=i + 1,
                                line_end=i + 1
                            ))
                            break
                
                self.detected_smells.append(SmellInstance(
                    smell_type=SmellType.SHOTGUN_SURGERY,
                    severity=SmellSeverity.MEDIUM,
                    description=f":  '{pattern_key.split('=')[0]}'  {len(files)} ",
                    locations=locations,
                    metadata={'files': files},
                    priority=40.0
                ))

    def _detect_feature_envy(self):
        all_files = []
        for module_path in self.analyzer._find_modules():
            all_files.extend(self.analyzer._find_python_files(module_path))
        
        for fp in all_files:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            local_names = self._collect_local_names(tree)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    external_access_count = 0
                    external_classes = set()
                    
                    for child in ast.walk(node):
                        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                            if child.value.id not in local_names:
                                external_access_count += 1
                                external_classes.add(child.value.id)
                    
                    if external_access_count > 5 and len(external_classes) >= 2:
                        line_start = node.lineno
                        with open(fp, 'r', encoding='utf-8') as f2:
                            lines = f2.readlines()
                        line_end = min(line_start + 20, len(lines))
                        
                        self.detected_smells.append(SmellInstance(
                            smell_type=SmellType.FEATURE_ENVY,
                            severity=SmellSeverity.LOW,
                            description=f":  '{func_name}'  {', '.join(external_classes)}",
                            locations=[CodeLocation(
                                file_path=fp,
                                line_start=line_start,
                                line_end=line_end
                            )],
                            metadata={
                                'function_name': func_name,
                                'external_access_count': external_access_count,
                                'external_classes': list(external_classes)
                            },
                            priority=30.0
                        ))

    def _collect_local_names(self, tree: ast.AST) -> Set[str]:
        # Collect names that are local to a file (class names and function arguments)
        local_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                local_names.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    local_names.add(arg.arg)
        return local_names

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor])
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            rec_stack.discard(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [node])
        
        return cycles

    def _normalize_snippet(self, snippet: str) -> str:
        lines = snippet.strip().split('\n')
        normalized = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                normalized.append(stripped)
        return '\n'.join(normalized)

    def _compute_function_complexity(self, node: ast.AST) -> int:
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
        
        return complexity

    def _prioritize_smells(self):
        severity_weights = {
            SmellSeverity.CRITICAL: 100,
            SmellSeverity.HIGH: 75,
            SmellSeverity.MEDIUM: 50,
            SmellSeverity.LOW: 25
        }
        
        type_weights = {
            SmellType.CYCLE_DEPENDENCY: 10,
            SmellType.DUPLICATE_CODE: 8,
            SmellType.GOD_MODULE: 7,
            SmellType.LONG_FUNCTION: 6,
            SmellType.DATA_CLUMP: 5,
            SmellType.SHOTGUN_SURGERY: 4,
            SmellType.FEATURE_ENVY: 3
        }
        
        for smell in self.detected_smells:
            base_score = severity_weights[smell.severity]
            type_score = type_weights[smell.smell_type] * 10
            location_factor = len(smell.locations) * 2
            smell.priority = base_score + type_score + location_factor + smell.priority

    def get_smells_by_type(self, smell_type: SmellType) -> List[SmellInstance]:
        return [s for s in self.detected_smells if s.smell_type == smell_type]

    def get_smells_by_severity(self, severity: SmellSeverity) -> List[SmellInstance]:
        return [s for s in self.detected_smells if s.severity == severity]

    def get_top_smells(self, n: int = 10) -> List[SmellInstance]:
        return sorted(self.detected_smells, key=lambda s: s.priority, reverse=True)[:n]