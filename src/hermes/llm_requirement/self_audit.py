from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import ast
import os
import sys
import re
import hashlib


class AuditIssueType(Enum):
    CODE_SMELL = "code_smell"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    SECURITY_VULNERABILITY = "security_vulnerability"
    FUNCTIONAL_GAP = "functional_gap"
    ARCHITECTURE_ISSUE = "architecture_issue"


class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AuditIssue:
    issue_type: AuditIssueType
    severity: IssueSeverity
    title: str
    description: str
    file_path: str
    line_number: int
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class AuditResult:
    issues: List[AuditIssue] = field(default_factory=list)
    total_files_scanned: int = 0
    total_issues_found: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_scanned": self.total_files_scanned,
            "total_issues_found": self.total_issues_found,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "issues": [issue.to_dict() for issue in self.issues]
        }


class SelfAuditor:
    LONG_FUNCTION_THRESHOLD = 50
    HIGH_COMPLEXITY_THRESHOLD = 10
    GOD_CLASS_THRESHOLD = 20
    DUPLICATE_MIN_LINES = 5
    
    def __init__(self, source_dir: Optional[str] = None):
        self.source_dir = source_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.issues: List[AuditIssue] = []
        self.scanned_files: Set[str] = set()
    
    def audit(self) -> AuditResult:
        self.issues = []
        self.scanned_files = set()
        
        print("[自我审计] 开始分析代码库...")
        
        for root, dirs, files in os.walk(self.source_dir):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'tests', 'generated_systems']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._audit_file(file_path)
        
        result = self._compile_result()
        self._print_summary(result)
        
        return result
    
    def _audit_file(self, file_path: str):
        if file_path in self.scanned_files:
            return
        
        self.scanned_files.add(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            self._detect_code_smells(tree, file_path, lines)
            self._detect_security_issues(tree, file_path, lines)
            self._detect_performance_issues(tree, file_path, lines)
            self._detect_functional_gaps(tree, file_path, lines)
            
        except Exception as e:
            pass
    
    def _detect_code_smells(self, tree: ast.AST, file_path: str, lines: List[str]):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = len(lines[node.lineno - 1:node.end_lineno])
                complexity = self._calculate_complexity(node)
                
                if func_lines > self.LONG_FUNCTION_THRESHOLD:
                    self.issues.append(AuditIssue(
                        issue_type=AuditIssueType.CODE_SMELL,
                        severity=IssueSeverity.MEDIUM,
                        title="过长函数",
                        description=f"函数 '{node.name}' 有 {func_lines} 行，超过阈值 {self.LONG_FUNCTION_THRESHOLD}",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestions=["拆分函数为多个小函数", "提取公共逻辑到辅助函数"]
                    ))
                
                if complexity > self.HIGH_COMPLEXITY_THRESHOLD:
                    self.issues.append(AuditIssue(
                        issue_type=AuditIssueType.CODE_SMELL,
                        severity=IssueSeverity.MEDIUM,
                        title="高复杂度函数",
                        description=f"函数 '{node.name}' 的圈复杂度为 {complexity}",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestions=["简化条件逻辑", "使用策略模式替代复杂 if-elif", "提取子逻辑"]
                    ))
            
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if method_count > self.GOD_CLASS_THRESHOLD:
                    self.issues.append(AuditIssue(
                        issue_type=AuditIssueType.CODE_SMELL,
                        severity=IssueSeverity.HIGH,
                        title="上帝类",
                        description=f"类 '{node.name}' 有 {method_count} 个方法，职责过多",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestions=["拆分类为多个职责单一的类", "使用组合替代继承"]
                    ))
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
            if isinstance(child, ast.Compare):
                complexity += len(child.ops)
        return complexity
    
    def _detect_security_issues(self, tree: ast.AST, file_path: str, lines: List[str]):
        security_patterns = [
            (r'\bexec\s*\(', "exec() 调用", IssueSeverity.CRITICAL),
            (r'\beval\s*\(', "eval() 调用", IssueSeverity.CRITICAL),
            (r'\bsubprocess\.Popen\s*\(', "subprocess.Popen 调用", IssueSeverity.HIGH),
            (r'\bsubprocess\.run\s*\(', "subprocess.run 调用", IssueSeverity.HIGH),
            (r'\binput\s*\(', "input() 调用", IssueSeverity.MEDIUM),
            (r'\bprint\s*\(', "print() 调用（建议使用 logging）", IssueSeverity.LOW),
            (r'password\s*=', "硬编码密码", IssueSeverity.CRITICAL),
            (r'secret\s*=', "硬编码密钥", IssueSeverity.CRITICAL),
            (r'token\s*=', "硬编码 token", IssueSeverity.CRITICAL),
        ]
        
        for i, line in enumerate(lines):
            for pattern, desc, severity in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.issues.append(AuditIssue(
                        issue_type=AuditIssueType.SECURITY_VULNERABILITY,
                        severity=severity,
                        title=desc,
                        description=f"第 {i+1} 行包含潜在安全风险: {desc}",
                        file_path=file_path,
                        line_number=i + 1,
                        suggestions=self._get_security_suggestions(desc)
                    ))
    
    def _get_security_suggestions(self, issue_desc: str) -> List[str]:
        if "exec" in issue_desc or "eval" in issue_desc:
            return ["避免使用 exec/eval，改用 AST 解析或白名单机制"]
        if "subprocess" in issue_desc:
            return ["使用安全的命令参数列表形式", "避免 shell=True", "验证输入参数"]
        if "password" in issue_desc or "secret" in issue_desc or "token" in issue_desc:
            return ["使用环境变量存储敏感信息", "使用密钥管理服务", "使用 .env 文件"]
        if "print" in issue_desc:
            return ["使用 logging 模块替代 print", "配置日志级别"]
        return []
    
    def _detect_performance_issues(self, tree: ast.AST, file_path: str, lines: List[str]):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.ListComp):
                        if isinstance(child.elt, (ast.Call, ast.Subscript)):
                            self.issues.append(AuditIssue(
                                issue_type=AuditIssueType.PERFORMANCE_BOTTLENECK,
                                severity=IssueSeverity.LOW,
                                title="嵌套列表推导",
                                description=f"函数 '{node.name}' 中包含复杂的列表推导",
                                file_path=file_path,
                                line_number=child.lineno,
                                suggestions=["考虑使用生成器表达式", "拆分复杂推导为多个步骤"]
                            ))
            
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.List):
                    self.issues.append(AuditIssue(
                        issue_type=AuditIssueType.PERFORMANCE_BOTTLENECK,
                        severity=IssueSeverity.LOW,
                        title="遍历大列表",
                        description=f"第 {node.lineno} 行遍历列表，可能效率低下",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestions=["考虑使用集合或字典查找", "使用生成器"]
                    ))
    
    def _detect_functional_gaps(self, tree: ast.AST, file_path: str, lines: List[str]):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('_'):
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        self.issues.append(AuditIssue(
                            issue_type=AuditIssueType.CODE_SMELL,
                            severity=IssueSeverity.LOW,
                            title="缺少文档字符串",
                            description=f"函数 '{node.name}' 缺少文档字符串",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggestions=["添加 docstring 说明函数用途和参数"]
                        ))
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.Try, ast.ExceptHandler)):
                        pass
    
    def _compile_result(self) -> AuditResult:
        result = AuditResult(
            issues=self.issues,
            total_files_scanned=len(self.scanned_files),
            total_issues_found=len(self.issues)
        )
        
        for issue in self.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                result.critical_count += 1
            elif issue.severity == IssueSeverity.HIGH:
                result.high_count += 1
            elif issue.severity == IssueSeverity.MEDIUM:
                result.medium_count += 1
            elif issue.severity == IssueSeverity.LOW:
                result.low_count += 1
        
        return result
    
    def _print_summary(self, result: AuditResult):
        print(f"[自我审计] 扫描完成: {result.total_files_scanned} 个文件")
        print(f"[自我审计] 发现问题:")
        print(f"  🔴 严重: {result.critical_count}")
        print(f"  🟠 高: {result.high_count}")
        print(f"  🟡 中: {result.medium_count}")
        print(f"  🟢 低: {result.low_count}")
        
        print(f"\n[自我审计] 主要问题:")
        for issue in sorted(result.issues, key=lambda i: i.severity.value)[:5]:
            print(f"  [{issue.severity.value}] {issue.title}")
            print(f"     {issue.description}")
            print(f"     文件: {os.path.basename(issue.file_path)}:{issue.line_number}")