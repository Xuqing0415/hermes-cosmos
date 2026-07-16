"""
CI Prover - Main entry point for CI/CD proof pipeline
"""

import time
from typing import List, Dict, Optional, Any
import structlog

from hermes.ci.types import (
    ChangedFunction,
    CIProofReport,
    PRInfo,
    DependencyGraph
)
from hermes.ci.change_analyzer import ChangeAnalyzer
from hermes.ci.proof_cache import ProofCache
from hermes.ci.incremental_prover import IncrementalProver
from hermes.ci.github_client import GitHubClient
from hermes.ci.ci_comment_formatter import CICommentFormatter

logger = structlog.get_logger()


class CIProver:
    """
    Main orchestrator for CI/CD proof pipeline.
    
    Workflow:
    1. Analyze changes (PR diff or local changes)
    2. Build dependency graph
    3. Perform incremental proof
    4. Generate report
    5. Post comment to PR (if configured)
    """
    
    def __init__(
        self,
        repo_path: str = ".",
        cache_path: str = "./proof_cache.db",
        solver: str = "z3",
        timeout: int = 30,
        certificate_dir: str = "./certificates"
    ):
        self.change_analyzer = ChangeAnalyzer(repo_path=repo_path)
        self.proof_cache = ProofCache(cache_path=cache_path)
        self.incremental_prover = IncrementalProver(
            proof_cache=self.proof_cache,
            solver=solver,
            timeout=timeout,
            certificate_dir=certificate_dir
        )
        self.github_client = GitHubClient()
        self.comment_formatter = CICommentFormatter()
        
        logger.info("CIProver initialized", repo_path=repo_path)
    
    def run_pr_proof(
        self,
        pr_info: PRInfo,
        base_sha: str,
        head_sha: str
    ) -> CIProofReport:
        """
        Run proof verification for a PR.
        
        Args:
            pr_info: PR information
            base_sha: Base commit SHA
            head_sha: Head commit SHA
        
        Returns:
            CIProofReport with results
        """
        logger.info("Running PR proof", pr_number=pr_info.pr_number)
        
        changed_functions = self.change_analyzer.analyze_pr(base_sha, head_sha)
        
        if not changed_functions:
            logger.info("No changed functions found")
            return CIProofReport(
                pr_number=pr_info.pr_number,
                commit_hash=head_sha
            )
        
        dependency_graph = self._build_dependency_graph()
        
        report = self.incremental_prover.prove_changed_functions(
            changed_functions,
            dependency_graph
        )
        
        report.pr_number = pr_info.pr_number
        report.commit_hash = head_sha
        
        return report
    
    def run_local_proof(self) -> CIProofReport:
        """
        Run proof verification for local changes.
        
        Returns:
            CIProofReport with results
        """
        logger.info("Running local proof")
        
        changed_functions = self.change_analyzer.analyze_local_changes()
        
        if not changed_functions:
            logger.info("No local changes found")
            return CIProofReport(
                pr_number=0,
                commit_hash="local"
            )
        
        dependency_graph = self._build_dependency_graph()
        
        report = self.incremental_prover.prove_changed_functions(
            changed_functions,
            dependency_graph
        )
        
        return report
    
    def run_push_proof(
        self,
        base_sha: str,
        head_sha: str
    ) -> CIProofReport:
        """
        Run proof verification for a push event.
        
        Args:
            base_sha: Base commit SHA (before push)
            head_sha: Head commit SHA (after push)
        
        Returns:
            CIProofReport with results
        """
        logger.info("Running push proof", base_sha=base_sha, head_sha=head_sha)
        
        changed_functions = self.change_analyzer.analyze_pr(base_sha, head_sha)
        
        if not changed_functions:
            logger.info("No changed functions found")
            return CIProofReport(
                pr_number=0,
                commit_hash=head_sha
            )
        
        dependency_graph = self._build_dependency_graph()
        
        report = self.incremental_prover.prove_changed_functions(
            changed_functions,
            dependency_graph
        )
        
        report.pr_number = 0
        report.commit_hash = head_sha
        
        return report
    
    def run_full_proof(self, files: List[str]) -> CIProofReport:
        """
        Run full proof verification for all functions in specified files.
        
        Args:
            files: List of Python files to analyze
        
        Returns:
            CIProofReport with results
        """
        logger.info("Running full proof", files=files)
        
        functions = []
        for filepath in files:
            cf = self._create_changed_file(filepath)
            if cf:
                functions.extend(cf)
        
        if not functions:
            logger.info("No functions found")
            return CIProofReport(
                pr_number=0,
                commit_hash="full"
            )
        
        report = self.incremental_prover.prove_all_functions(functions)
        
        return report
    
    def _create_changed_file(self, filepath: str) -> Optional[List[ChangedFunction]]:
        """Create ChangedFunction objects from a file path"""
        if not filepath.endswith(".py"):
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            import ast
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    import hashlib
                    line_start = node.lineno
                    line_end = node.end_lineno
                    source_lines = content.split("\n")
                    func_source = "\n".join(source_lines[line_start - 1:line_end])
                    signature = hashlib.sha256(func_source.encode()).hexdigest()[:16]
                    
                    functions.append(ChangedFunction(
                        name=node.name,
                        filename=filepath,
                        line_start=line_start,
                        line_end=line_end,
                        signature=signature
                    ))
            
            return functions
        except Exception as e:
            logger.error("Failed to parse file", filepath=filepath, error=str(e))
            return None
    
    def _build_dependency_graph(self) -> Optional[DependencyGraph]:
        """Build a dependency graph for the repository"""
        try:
            import glob
            python_files = glob.glob("**/*.py", recursive=True)
            return self.change_analyzer.build_dependency_graph(python_files)
        except Exception as e:
            logger.warning("Failed to build dependency graph", error=str(e))
            return None
    
    def post_pr_comment(self, pr_info: PRInfo, report: CIProofReport) -> Optional[Dict[str, Any]]:
        """
        Post a comment to the PR with proof results.
        
        Args:
            pr_info: PR information
            report: CIProofReport to post
        
        Returns:
            Comment response from GitHub API
        """
        logger.info("Posting PR comment", pr_number=pr_info.pr_number)
        
        self.github_client.set_repo(pr_info.owner, pr_info.repo)
        
        body = self.comment_formatter.format_report(report)
        
        return self.github_client.post_pr_comment(pr_info.pr_number, body)
    
    def update_pr_comment(
        self,
        pr_info: PRInfo,
        comment_id: int,
        report: CIProofReport
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing PR comment.
        
        Args:
            pr_info: PR information
            comment_id: ID of the comment to update
            report: CIProofReport to post
        
        Returns:
            Updated comment response from GitHub API
        """
        logger.info("Updating PR comment", pr_number=pr_info.pr_number, comment_id=comment_id)
        
        self.github_client.set_repo(pr_info.owner, pr_info.repo)
        
        body = self.comment_formatter.format_report(report)
        
        return self.github_client.update_pr_comment(comment_id, body)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.proof_cache.get_stats()
    
    def invalidate_cache(self, function_name: Optional[str] = None, filename: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            function_name: If specified, invalidate only this function
            filename: If specified, invalidate all functions in this file
        """
        if function_name:
            self.proof_cache.invalidate(function_name)
        elif filename:
            self.proof_cache.invalidate_file(filename)
        else:
            logger.warning("No function name or filename provided for invalidation")
    
    def run_pipeline(
        self,
        pr_info: Optional[PRInfo] = None,
        base_sha: Optional[str] = None,
        head_sha: Optional[str] = None,
        post_comment: bool = True
    ) -> CIProofReport:
        """
        Run the complete CI pipeline.
        
        Args:
            pr_info: PR information (if running for a PR)
            base_sha: Base commit SHA
            head_sha: Head commit SHA
            post_comment: Whether to post comment to PR
        
        Returns:
            CIProofReport with results
        """
        start_time = time.time()
        logger.info("Starting CI proof pipeline")
        
        if pr_info and base_sha and head_sha:
            report = self.run_pr_proof(pr_info, base_sha, head_sha)
        else:
            report = self.run_local_proof()
        
        report.total_duration = time.time() - start_time
        
        if post_comment and pr_info:
            self.post_pr_comment(pr_info, report)
        
        logger.info(
            "CI proof pipeline complete",
            proven=report.total_proven,
            disproven=report.total_disproven,
            duration=report.total_duration
        )
        
        return report
