"""
Proof Certificate - Stores proof results and reproducible scripts
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any, List
import structlog
import os

from hermes.neural_symbolic.types import ProofResult, ProofStatus, GeneratedTest

logger = structlog.get_logger()


class ProofCertificate:
    """
    Stores and manages proof certificates.
    
    A proof certificate contains:
    - The original SMT expression
    - The proof result (proven/disproven/unknown)
    - Timestamps and metadata
    - Reproducibility script
    """
    
    def __init__(self, certificate_id: Optional[str] = None):
        self.id = certificate_id or self._generate_id()
        self.results: List[Dict[str, Any]] = []
        self.tests: List[GeneratedTest] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def _generate_id(self) -> str:
        """Generate a unique certificate ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_val = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"cert_{timestamp}_{hash_val}"
    
    def add_proof_result(self, result: ProofResult, target_info: Dict[str, Any] = None):
        """
        Add a proof result to the certificate.
        
        Args:
            result: ProofResult object
            target_info: Additional information about the proof target
        """
        entry = {
            "id": result.smt_expression[:32] + "..." if len(result.smt_expression) > 32 else result.smt_expression,
            "smt_expression": result.smt_expression,
            "status": result.status.value,
            "model": result.model,
            "certificate": result.certificate,
            "duration": result.duration,
            "error_message": result.error_message,
            "target_info": target_info or {},
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(entry)
        self.updated_at = datetime.now().isoformat()
        logger.info("Proof result added to certificate", certificate_id=self.id, status=result.status.value)
    
    def add_test(self, test: GeneratedTest):
        """
        Add a generated test to the certificate.
        
        Args:
            test: GeneratedTest object
        """
        self.tests.append(test)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to dictionary"""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "results": self.results,
            "tests": [
                {
                    "id": t.id,
                    "function_name": t.function_name,
                    "inputs": t.inputs,
                    "expected_output": t.expected_output,
                    "assertion": t.assertion,
                    "description": t.description
                }
                for t in self.tests
            ],
            "statistics": self._compute_statistics()
        }
    
    def _compute_statistics(self) -> Dict[str, int]:
        """Compute statistics from results"""
        stats = {
            "total": len(self.results),
            "proven": 0,
            "disproven": 0,
            "unknown": 0,
            "timeout": 0
        }
        
        for result in self.results:
            status = result["status"]
            if status == ProofStatus.PROVEN.value:
                stats["proven"] += 1
            elif status == ProofStatus.DISPROVEN.value:
                stats["disproven"] += 1
            elif status == ProofStatus.UNKNOWN.value:
                stats["unknown"] += 1
            elif status == ProofStatus.TIMEOUT.value:
                stats["timeout"] += 1
        
        return stats
    
    def to_json(self, indent: int = 2) -> str:
        """Convert certificate to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, directory: str = "./certificates") -> str:
        """
        Save certificate to a file.
        
        Args:
            directory: Directory to save the certificate
        
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.id}.json")
        
        with open(filepath, "w") as f:
            f.write(self.to_json())
        
        logger.info("Certificate saved", filepath=filepath)
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ProofCertificate":
        """
        Load certificate from a file.
        
        Args:
            filepath: Path to the certificate file
        
        Returns:
            ProofCertificate instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        cert = cls(certificate_id=data["id"])
        cert.created_at = data["created_at"]
        cert.updated_at = data["updated_at"]
        cert.results = data.get("results", [])
        
        for test_data in data.get("tests", []):
            test = GeneratedTest(
                id=test_data["id"],
                function_name=test_data["function_name"],
                inputs=test_data["inputs"],
                expected_output=test_data.get("expected_output"),
                assertion=test_data.get("assertion"),
                description=test_data.get("description")
            )
            cert.add_test(test)
        
        logger.info("Certificate loaded", filepath=filepath)
        return cert
    
    def generate_reproducibility_script(self) -> str:
        """
        Generate a Python script that can reproduce the proof.
        
        Returns:
            Python script as string
        """
        script = f'''#!/usr/bin/env python3
"""
Reproducibility script for proof certificate {self.id}
Generated: {self.created_at}
"""

import sys
from hermes.neural_symbolic.theorem_prover import TheoremProver
from hermes.neural_symbolic.proof_obligation import ProofObligationGenerator

def reproduce():
    prover = TheoremProver(timeout=30)
    generator = ProofObligationGenerator()
    
    results = []
    for i, entry in enumerate({self.results}, 1):
        print(f"Proof {{i}}/{{len({self.results})}}: {{entry['status']}}")
        print(f"  Expression: {{entry['smt_expression'][:80]}}...")
        
        result = prover.prove(entry['smt_expression'])
        results.append(result)
        
        if result.status == 'proven':
            print(f"   Proven in {{result.duration:.2f}}s")
        elif result.status == 'disproven':
            print(f"   Disproven, model: {{result.model}}")
        else:
            print(f"  ? {{result.status}}")
    
    return results

if __name__ == "__main__":
    reproduce()
'''
        return script
    
    def save_reproducibility_script(self, directory: str = "./certificates") -> str:
        """
        Save reproducibility script to a file.
        
        Args:
            directory: Directory to save the script
        
        Returns:
            Path to the saved script
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.id}_reproduce.py")
        
        with open(filepath, "w") as f:
            f.write(self.generate_reproducibility_script())
        
        os.chmod(filepath, 0o755)
        logger.info("Reproducibility script saved", filepath=filepath)
        return filepath
