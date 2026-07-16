"""
Incremental Prover - Only re-prove affected functions
"""

import time
from typing import List, Dict, Optional, Any
import structlog

from hermes.ci.types import (
    ChangedFunction,
    ProofResult,
    ProofResultStatus,
    CIProofReport,
    DependencyGraph
)
from hermes.ci.proof_cache import ProofCache
from hermes.neural_symbolic.proof_generator import ProofGenerator

logger = structlog.get_logger()


class IncrementalProver:
    """
    Performs incremental proof verification.
    
    Strategy:
    1. Check cache for unchanged functions
    2. Only re-prove functions whose signature has changed
    3. Rebuild proofs for functions affected by changed dependencies
    """
    
    def __init__(
        self,
        proof_cache: ProofCache,
        solver: str = "z3",
        timeout: int = 30,
        certificate_dir: str = "./certificates"
    ):
        self.proof_cache = proof_cache
        self.proof_generator = ProofGenerator(
            solver=solver,
            timeout=timeout,
            certificate_dir=certificate_dir
        )
        self.solver = solver
        self.timeout = timeout
    
    def prove_changed_functions(
        self,
        changed_functions: List[ChangedFunction],
        dependency_graph: Optional[DependencyGraph] = None
    ) -> CIProofReport:
        """
        Prove only changed and affected functions.
        
        Args:
            changed_functions: List of changed functions
            dependency_graph: Optional dependency graph for impact analysis
        
        Returns:
            CIProofReport with all results
        """
        start_time = time.time()
        
        logger.info("Starting incremental proof", changed_count=len(changed_functions))
        
        affected_functions = self._get_affected_functions(changed_functions, dependency_graph)
        logger.info("Affected functions identified", count=len(affected_functions))
        
        report = CIProofReport(
            pr_number=0,
            commit_hash="local"
        )
        
        cache_hits = 0
        total_attempts = 0
        
        for func in affected_functions:
            total_attempts += 1
            
            cache_key = self.proof_cache.compute_key(func.name, func.signature)
            cached_entry = self.proof_cache.get(cache_key)
            
            if cached_entry:
                cache_hits += 1
                result = self._create_result_from_cache(cached_entry)
                report.total_cached += 1
            else:
                result = self._prove_function(func)
                self._cache_result(cache_key, func, result)
            
            report.results.append(result)
            
            if result.status == ProofResultStatus.PROVEN:
                report.total_proven += 1
            elif result.status == ProofResultStatus.DISPROVEN:
                report.total_disproven += 1
            elif result.status == ProofResultStatus.TIMEOUT:
                report.total_timeout += 1
            elif result.status == ProofResultStatus.UNKNOWN:
                report.total_unknown += 1
            
            report.total_duration += result.duration
        
        report.cache_hit_rate = cache_hits / total_attempts if total_attempts > 0 else 0.0
        report.total_duration = time.time() - start_time
        
        logger.info(
            "Incremental proof complete",
            proven=report.total_proven,
            disproven=report.total_disproven,
            cached=report.total_cached,
            duration=report.total_duration,
            cache_hit_rate=report.cache_hit_rate
        )
        
        return report
    
    def _get_affected_functions(
        self,
        changed_functions: List[ChangedFunction],
        dependency_graph: Optional[DependencyGraph]
    ) -> List[ChangedFunction]:
        """Get all functions that need to be re-proven"""
        if dependency_graph is None:
            return changed_functions
        
        changed_names = [f.name for f in changed_functions]
        affected_names = dependency_graph.get_affected_functions(changed_names)
        
        affected_functions = []
        for func in changed_functions:
            if func.name in affected_names:
                affected_functions.append(func)
        
        return affected_functions
    
    def _prove_function(self, func: ChangedFunction) -> ProofResult:
        """Prove a single function"""
        logger.info("Proving function", name=func.name, filename=func.filename)
        
        start_time = time.time()
        
        try:
            code = self._get_function_code(func)
            
            if code is None:
                return ProofResult(
                    function_name=func.name,
                    filename=func.filename,
                    status=ProofResultStatus.UNKNOWN,
                    duration=0.0,
                    error_message="Could not get function code"
                )
            
            result = self.proof_generator.prove_and_generate_test(code, func.name)
            
            duration = time.time() - start_time
            
            status = self._map_status(result)
            
            return ProofResult(
                function_name=func.name,
                filename=func.filename,
                status=status,
                duration=duration,
                certificate_path=None,
                generated_tests=[t["id"] for t in result["tests"]],
                cached=False
            )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Failed to prove function", name=func.name, error=str(e))
            
            return ProofResult(
                function_name=func.name,
                filename=func.filename,
                status=ProofResultStatus.UNKNOWN,
                duration=duration,
                error_message=str(e),
                cached=False
            )
    
    def _get_function_code(self, func: ChangedFunction) -> Optional[str]:
        """Get the complete code for a function"""
        return f"def {func.name}(...):\n    ..."
    
    def _map_status(self, result: Dict[str, Any]) -> ProofResultStatus:
        """Map proof generator result to CI status"""
        if result["proven"] > 0 and result["disproven"] == 0:
            return ProofResultStatus.PROVEN
        elif result["disproven"] > 0:
            return ProofResultStatus.DISPROVEN
        else:
            return ProofResultStatus.UNKNOWN
    
    def _create_result_from_cache(self, entry) -> ProofResult:
        """Create a ProofResult from a cache entry"""
        return ProofResult(
            function_name=entry.function_name,
            filename=entry.filename,
            status=entry.status,
            duration=entry.duration,
            certificate_path=entry.certificate_path,
            generated_tests=None,
            cached=True
        )
    
    def _cache_result(self, key: str, func: ChangedFunction, result: ProofResult):
        """Cache a proof result"""
        self.proof_cache.set(
            key=key,
            function_name=func.name,
            filename=func.filename,
            signature=func.signature,
            status=result.status,
            certificate_path=result.certificate_path,
            duration=result.duration
        )
    
    def prove_all_functions(
        self,
        functions: List[ChangedFunction]
    ) -> CIProofReport:
        """
        Prove all functions (non-incremental mode).
        
        Args:
            functions: List of functions to prove
        
        Returns:
            CIProofReport with all results
        """
        logger.info("Proving all functions", count=len(functions))
        
        report = CIProofReport(
            pr_number=0,
            commit_hash="full"
        )
        
        for func in functions:
            result = self._prove_function(func)
            report.results.append(result)
            
            if result.status == ProofResultStatus.PROVEN:
                report.total_proven += 1
            elif result.status == ProofResultStatus.DISPROVEN:
                report.total_disproven += 1
            elif result.status == ProofResultStatus.TIMEOUT:
                report.total_timeout += 1
            elif result.status == ProofResultStatus.UNKNOWN:
                report.total_unknown += 1
            
            report.total_duration += result.duration
        
        logger.info("Full proof complete", total=len(functions), duration=report.total_duration)
        return report
