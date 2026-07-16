"""
Proof Generator - Main entry point for neural-symbolic proof generation
"""

import time
from typing import List, Dict, Optional, Any
import structlog

from hermes.neural_symbolic.types import (
    ProofResult, 
    ProofTarget, 
    ProofReport, 
    GeneratedTest,
    ProofStatus
)
from hermes.neural_symbolic.theorem_prover import TheoremProver
from hermes.neural_symbolic.proof_obligation import ProofObligationGenerator
from hermes.neural_symbolic.certificate import ProofCertificate
from hermes.neural_symbolic.target_predictor import NeuralProofTargetPredictor
from hermes.neural_symbolic.test_generator import ProofGuidedTestGenerator

logger = structlog.get_logger()


class ProofGenerator:
    """
    Main orchestrator for neural-symbolic proof generation.
    
    Workflow:
    1. Generate proof targets from code
    2. Predict high-value targets using neural model
    3. Generate verification conditions
    4. Prove each target using theorem prover
    5. Generate tests for disproven/unknown targets
    6. Create proof certificate
    """
    
    def __init__(
        self,
        solver: str = "z3",
        timeout: int = 30,
        top_n: int = 10,
        certificate_dir: str = "./certificates"
    ):
        self.prover = TheoremProver(solver=solver, timeout=timeout)
        self.obligation_generator = ProofObligationGenerator()
        self.target_predictor = NeuralProofTargetPredictor()
        self.test_generator = ProofGuidedTestGenerator()
        self.certificate_dir = certificate_dir
        self.top_n = top_n
        
        logger.info("ProofGenerator initialized", solver=solver, timeout=timeout)
    
    def generate_proof_report(
        self,
        code: str,
        function_name: str,
        precondition: Optional[str] = None,
        postcondition: Optional[str] = None
    ) -> ProofReport:
        """
        Generate a complete proof report for a function.
        
        Args:
            code: Python source code
            function_name: Target function name
            precondition: Optional precondition for verification
            postcondition: Optional postcondition for verification
        
        Returns:
            ProofReport with all results
        """
        start_time = time.time()
        
        logger.info("Starting proof generation", function_name=function_name)
        
        targets = self._generate_proof_targets(code, function_name)
        
        if not targets:
            logger.warning("No proof targets generated", function_name=function_name)
            return ProofReport(
                proven_count=0,
                disproven_count=0,
                unknown_count=0,
                timeout_count=0,
                proven_targets=[],
                disproven_targets=[],
                generated_tests=[],
                total_duration=time.time() - start_time
            )
        
        top_targets = self._predict_top_targets(targets)
        
        results = self._prove_targets(top_targets, precondition, postcondition)
        
        certificate = self._create_certificate(results, top_targets)
        
        tests = self._generate_tests(results, top_targets)
        
        for test in tests:
            certificate.add_test(test)
        
        certificate_path = certificate.save(self.certificate_dir)
        certificate.save_reproducibility_script(self.certificate_dir)
        
        report = self._create_report(results, top_targets, tests, start_time)
        
        logger.info(
            "Proof generation complete",
            proven=report.proven_count,
            disproven=report.disproven_count,
            unknown=report.unknown_count,
            tests=len(report.generated_tests),
            duration=report.total_duration
        )
        
        return report
    
    def _generate_proof_targets(self, code: str, function_name: str) -> List[ProofTarget]:
        """Generate proof targets from code"""
        targets = self.obligation_generator.generate_proof_targets(code, function_name)
        logger.info("Proof targets generated", count=len(targets))
        return targets
    
    def _predict_top_targets(self, targets: List[ProofTarget]) -> List[ProofTarget]:
        """Predict top targets using neural model"""
        top_targets = self.target_predictor.predict_top_targets(targets, self.top_n)
        logger.info("Top targets selected", count=len(top_targets))
        return top_targets
    
    def _prove_targets(
        self,
        targets: List[ProofTarget],
        precondition: Optional[str] = None,
        postcondition: Optional[str] = None
    ) -> List[ProofResult]:
        """Prove all targets using theorem prover"""
        results = []
        
        for i, target in enumerate(targets, 1):
            logger.info(f"Proving target {i}/{len(targets)}", id=target.id)
            
            vc = self.obligation_generator.generate_from_path_condition(
                target.path_condition,
                precondition,
                postcondition
            )
            
            result = self.prover.prove(vc.smt_formula)
            results.append(result)
            
            status_str = f" PROVEN" if result.status == ProofStatus.PROVEN else \
                         f" DISPROVEN" if result.status == ProofStatus.DISPROVEN else \
                         f"? UNKNOWN"
            logger.info(f"  {status_str} ({result.duration:.2f}s)")
        
        return results
    
    def _create_certificate(
        self,
        results: List[ProofResult],
        targets: List[ProofTarget]
    ) -> ProofCertificate:
        """Create proof certificate from results"""
        certificate = ProofCertificate()
        
        for result, target in zip(results, targets):
            target_info = {
                "function_name": target.function_name,
                "risk_score": target.risk_score,
                "priority": target.priority
            }
            certificate.add_proof_result(result, target_info)
        
        logger.info("Proof certificate created", id=certificate.id)
        return certificate
    
    def _generate_tests(
        self,
        results: List[ProofResult],
        targets: List[ProofTarget]
    ) -> List[GeneratedTest]:
        """Generate tests from disproven/unknown results"""
        tests = []
        
        for result, target in zip(results, targets):
            test = self.test_generator.generate_from_proof_result(result, target)
            if test:
                tests.append(test)
        
        logger.info("Tests generated", count=len(tests))
        return tests
    
    def _create_report(
        self,
        results: List[ProofResult],
        targets: List[ProofTarget],
        tests: List[GeneratedTest],
        start_time: float
    ) -> ProofReport:
        """Create comprehensive proof report"""
        proven_count = sum(1 for r in results if r.status == ProofStatus.PROVEN)
        disproven_count = sum(1 for r in results if r.status == ProofStatus.DISPROVEN)
        unknown_count = sum(1 for r in results if r.status == ProofStatus.UNKNOWN)
        timeout_count = sum(1 for r in results if r.status == ProofStatus.TIMEOUT)
        
        proven_targets = [
            t for r, t in zip(results, targets) if r.status == ProofStatus.PROVEN
        ]
        disproven_targets = [
            t for r, t in zip(results, targets) if r.status == ProofStatus.DISPROVEN
        ]
        
        return ProofReport(
            proven_count=proven_count,
            disproven_count=disproven_count,
            unknown_count=unknown_count,
            timeout_count=timeout_count,
            proven_targets=proven_targets,
            disproven_targets=disproven_targets,
            generated_tests=tests,
            total_duration=time.time() - start_time
        )
    
    def prove_and_generate_test(
        self,
        code: str,
        function_name: str
    ) -> Dict[str, Any]:
        """
        Quick proof and test generation.
        
        Args:
            code: Python source code
            function_name: Target function name
        
        Returns:
            Dictionary with results and generated tests
        """
        report = self.generate_proof_report(code, function_name)
        
        return {
            "proven": report.proven_count,
            "disproven": report.disproven_count,
            "unknown": report.unknown_count,
            "tests": [
                {
                    "id": t.id,
                    "function_name": t.function_name,
                    "inputs": t.inputs,
                    "assertion": t.assertion,
                    "description": t.description
                }
                for t in report.generated_tests
            ],
            "test_code": self.test_generator.generate_test_module(report.generated_tests),
            "duration": report.total_duration
        }
    
    def verify_division_safety(self, code: str, function_name: str) -> Dict[str, Any]:
        """
        Specialized verification for division safety.
        
        Args:
            code: Python source code
            function_name: Target function name
        
        Returns:
            Dictionary with verification results
        """
        targets = self.obligation_generator.generate_proof_targets(code, function_name)
        div_targets = [t for t in targets if "/" in t.path_condition.condition]
        
        if not div_targets:
            return {"result": "No division operations found", "safe": True, "proof_results": []}
        
        results = []
        for target in div_targets:
            result = self.prover.verify_no_div_by_zero("b")
            results.append(result)
        
        all_proven = all(r.status == ProofStatus.PROVEN for r in results)
        
        tests = []
        if not all_proven:
            tests = self._generate_tests(results, div_targets)
        
        return {
            "result": "SAFE" if all_proven else "UNSAFE",
            "safe": all_proven,
            "proof_results": [r.status.value for r in results],
            "generated_tests": len(tests),
            "test_code": self.test_generator.generate_test_module(tests)
        }
