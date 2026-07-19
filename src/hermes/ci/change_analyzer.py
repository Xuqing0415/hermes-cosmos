"""
Change Analyzer - Parses PR diff to identify modified functions
"""

import ast
import hashlib
import subprocess
import tempfile
import os
from typing import List, Dict, Optional, Any
import structlog

from hermes.ci.types import ChangedFile, ChangedFunction, DependencyGraph

logger = structlog.get_logger()


class ChangeAnalyzer:
    """
    Analyzes changes between commits to identify affected functions.
    
    Uses git diff and AST analysis to:
    1. Identify changed files
    2. Extract modified functions
    3. Build dependency graph
    4. Compute function signatures
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
    
    def analyze_pr(self, base_sha: str, head_sha: str) -> List[ChangedFunction]:
        """
        Analyze changes between two commits (base and head of a PR).
        
        Args:
            base_sha: Base commit SHA
            head_sha: Head commit SHA
        
        Returns:
            List of ChangedFunction objects
        """
        logger.info("Analyzing PR changes", base_sha=base_sha, head_sha=head_sha)
        
        changed_files = self._get_changed_files(base_sha, head_sha)
        logger.info("Changed files identified", count=len(changed_files))
        
        changed_functions = []
        for cf in changed_files:
            if cf.filename.endswith(".py"):
                functions = self._extract_changed_functions(cf)
                changed_functions.extend(functions)
        
        logger.info("Changed functions identified", count=len(changed_functions))
        return changed_functions
    
    def _get_changed_files(self, base_sha: str, head_sha: str) -> List[ChangedFile]:
        """Get list of files changed between two commits"""
        try:
            cmd = [
                "git", "diff", "--numstat",
                f"{base_sha}...{head_sha}"
            ]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                
                parts = line.split("\t")
                if len(parts) >= 3:
                    additions = int(parts[0])
                    deletions = int(parts[1])
                    filename = parts[2]
                    changes = additions + deletions
                    
                    files.append(ChangedFile(
                        filename=filename,
                        additions=additions,
                        deletions=deletions,
                        changes=changes,
                        status="modified"
                    ))
            
            return files
        except subprocess.CalledProcessError as e:
            logger.error("Failed to get changed files", error=str(e))
            return []
    
    def _extract_changed_functions(self, changed_file: ChangedFile) -> List[ChangedFunction]:
        """Extract functions from a changed Python file"""
        filepath = os.path.join(self.repo_path, changed_file.filename)
        
        if not os.path.exists(filepath):
            logger.warning("File not found", filepath=filepath)
            return []
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func = self._parse_function(node, changed_file.filename, content)
                    functions.append(func)
                elif isinstance(node, ast.AsyncFunctionDef):
                    func = self._parse_function(node, changed_file.filename, content)
                    functions.append(func)
            
            return functions
        except Exception as e:
            logger.error("Failed to parse file", filepath=filepath, error=str(e))
            return []
    
    def _parse_function(
        self,
        node: ast.FunctionDef,
        filename: str,
        content: str
    ) -> ChangedFunction:
        """Parse a function node into ChangedFunction"""
        line_start = node.lineno
        line_end = node.end_lineno
        
        source_lines = content.split("\n")
        func_source = "\n".join(source_lines[line_start - 1:line_end])
        signature = self._compute_function_signature(func_source)
        
        return ChangedFunction(
            name=node.name,
            filename=filename,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            affected_dependencies=[]
        )
    
    def _compute_function_signature(self, source_code: str) -> str:
        """Compute a hash signature for a function"""
        return hashlib.sha256(source_code.encode()).hexdigest()[:16]
    
    def build_dependency_graph(self, files: List[str]) -> DependencyGraph:
        """
        Build a dependency graph from Python files.
        
        Args:
            files: List of Python file paths
        
        Returns:
            DependencyGraph
        """
        functions = []
        dependencies: Dict[str, List[str]] = {}
        
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                tree = ast.parse(content)
                module_name = os.path.basename(filepath).replace(".py", "")
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        func_name = f"{module_name}.{node.name}"
                        functions.append(func_name)
                        dependencies[func_name] = self._extract_dependencies(node, module_name)
            except Exception as e:
                logger.error("Failed to parse file for dependencies", filepath=filepath, error=str(e))
        
        return DependencyGraph(functions=functions, dependencies=dependencies)
    
    def _extract_dependencies(self, node: ast.FunctionDef, module_name: str) -> List[str]:
        """Extract function dependencies from a function node"""
        deps = []
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                deps.append(f"{module_name}.{subnode.func.id}")
        
        return list(set(deps))
    
    def analyze_local_changes(self) -> List[ChangedFunction]:
        """Analyze uncommitted local changes"""
        logger.info("Analyzing local changes")
        
        try:
            cmd = ["git", "diff", "--numstat"]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                
                parts = line.split("\t")
                if len(parts) >= 3:
                    additions = int(parts[0])
                    deletions = int(parts[1])
                    filename = parts[2]
                    files.append(ChangedFile(
                        filename=filename,
                        additions=additions,
                        deletions=deletions,
                        changes=additions + deletions,
                        status="modified"
                    ))
            
            changed_functions = []
            for cf in files:
                if cf.filename.endswith(".py"):
                    functions = self._extract_changed_functions(cf)
                    changed_functions.extend(functions)
            
            return changed_functions
        except subprocess.CalledProcessError as e:
            logger.error("Failed to analyze local changes", error=str(e))
            return []
    
    def get_function_source(self, filename: str, function_name: str) -> Optional[str]:
        """
        Get the source code of a specific function.
        
        Args:
            filename: Path to the Python file
            function_name: Name of the function
        
        Returns:
            Source code as string, or None
        """
        filepath = os.path.join(self.repo_path, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        lines = content.split("\n")
                        return "\n".join(lines[node.lineno - 1:node.end_lineno])
            
            return None
        except Exception as e:
            logger.error("Failed to get function source", error=str(e))
            return None
