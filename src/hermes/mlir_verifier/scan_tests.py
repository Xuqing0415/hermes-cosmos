import os
import re
import json
import argparse
from typing import List, Dict, Optional, Any, Tuple

from hermes.mlir_verifier.symbolic_executor import MLIRSymbolicExecutor
from hermes.mlir_verifier.types import VerificationStatus


SUPPORTED_DIALECTS = {"func", "arith", "cf", "memref", "builtin"}
UNSUPPORTED_DIALECTS = {
    "scf", "linalg", "vector", "tensor", "affine", "gpu", "gpu.launch",
    "async", "omp", "sparse_tensor", "bufferization", "transform", "llvm",
    "nvvm", "rocm", "amdgpu", "arm_sve", "arm_neon", "x86vector",
    "gpu.memory", "gpu.barrier", "gpu.wait", "gpu.shmem",
}


def detect_dialects(mlir_code: str) -> Tuple[List[str], List[str]]:
    dialect_pattern = r'(?<!")([a-zA-Z_][a-zA-Z0-9_.]*)\.'
    matches = re.findall(dialect_pattern, mlir_code)
    
    used_dialects = set()
    for match in matches:
        dialect = match.split('.')[0] if '.' in match else match
        if dialect:
            used_dialects.add(dialect)
    
    supported = [d for d in used_dialects if d in SUPPORTED_DIALECTS]
    unsupported = [d for d in used_dialects if d in UNSUPPORTED_DIALECTS]
    
    return sorted(supported), sorted(unsupported)


def is_file_supported(mlir_code: str) -> bool:
    _, unsupported = detect_dialects(mlir_code)
    return len(unsupported) == 0


def extract_functions(mlir_code: str) -> List[str]:
    func_pattern = r'(func\.func\s+@[\w]+\s*\([^)]*\)\s*->\s*[^\{]*\{[^}]+\})'
    matches = re.findall(func_pattern, mlir_code, re.DOTALL)
    return matches


def scan_file(filepath: str, executor: MLIRSymbolicExecutor) -> Dict[str, Any]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            mlir_code = f.read()
    except Exception as e:
        return {
            "filepath": filepath,
            "error": f"Cannot read file: {e}",
            "supported": False
        }
    
    supported_dialects, unsupported_dialects = detect_dialects(mlir_code)
    
    if not is_file_supported(mlir_code):
        return {
            "filepath": filepath,
            "supported": False,
            "supported_dialects": supported_dialects,
            "unsupported_dialects": unsupported_dialects,
            "skipped_reason": "Contains unsupported dialects"
        }
    
    functions = extract_functions(mlir_code)
    
    results = {
        "filepath": filepath,
        "supported": True,
        "supported_dialects": supported_dialects,
        "unsupported_dialects": unsupported_dialects,
        "total_functions": len(functions),
        "functions": []
    }
    
    for func_code in functions:
        func_name_match = re.search(r'func\.func\s+@(\w+)', func_code)
        func_name = func_name_match.group(1) if func_name_match else "unknown"
        
        try:
            cfg = executor.parse_mlir_to_cfg(func_code)
            verification_result = executor.verify_function(func_code)
            
            func_result = {
                "name": func_name,
                "status": verification_result.status.value,
                "errors": [{
                    "type": e.error_type.value,
                    "message": e.message,
                    "location": e.location
                } for e in verification_result.errors]
            }
            
            if verification_result.path_conditions:
                func_result["path_conditions"] = len(verification_result.path_conditions)
            
            results["functions"].append(func_result)
            
        except Exception as e:
            results["functions"].append({
                "name": func_name,
                "status": "error",
                "error": str(e)
            })
    
    return results


def scan_directory(root_dir: str, executor: MLIRSymbolicExecutor,
                   max_files: int = 1000, verbose: bool = False) -> List[Dict[str, Any]]:
    all_results = []
    file_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        for filename in filenames:
            if not filename.endswith('.mlir'):
                continue
            
            if file_count >= max_files:
                break
            
            filepath = os.path.join(dirpath, filename)
            
            if verbose:
                print(f"Scanning: {filepath}")
            
            result = scan_file(filepath, executor)
            all_results.append(result)
            file_count += 1
        
        if file_count >= max_files:
            break
    
    return all_results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_files = len(results)
    supported_files = sum(1 for r in results if r.get("supported", False))
    unsupported_files = total_files - supported_files
    
    total_functions = sum(r.get("total_functions", 0) for r in results if r.get("supported", False))
    
    status_counts = {}
    error_counts = {}
    
    for result in results:
        if not result.get("supported", False):
            continue
        
        for func_result in result.get("functions", []):
            status = func_result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            for error in func_result.get("errors", []):
                error_type = error.get("type", "unknown")
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    suspicious_cases = []
    for result in results:
        if not result.get("supported", False):
            continue
        
        for func_result in result.get("functions", []):
            if func_result.get("status") == VerificationStatus.BOUNDARY_VIOLATION.value:
                suspicious_cases.append({
                    "filepath": result["filepath"],
                    "function": func_result["name"],
                    "errors": func_result["errors"]
                })
    
    return {
        "total_files": total_files,
        "supported_files": supported_files,
        "unsupported_files": unsupported_files,
        "total_functions": total_functions,
        "status_distribution": status_counts,
        "error_distribution": error_counts,
        "suspicious_cases": suspicious_cases,
        "suspicious_count": len(suspicious_cases)
    }


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any], output_file: str):
    output = {
        "summary": summary,
        "timestamp": __import__('time').time(),
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Scan MLIR test files for boundary violations")
    parser.add_argument("directory", help="Root directory containing MLIR files")
    parser.add_argument("--max-files", type=int, default=1000, help="Maximum number of files to scan")
    parser.add_argument("--output", default="scan_results.json", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    
    args = parser.parse_args()
    
    print(f"Starting MLIR test scan of {args.directory}")
    print(f"Max files: {args.max_files}")
    
    executor = MLIRSymbolicExecutor()
    
    results = scan_directory(args.directory, executor, max_files=args.max_files, verbose=args.verbose)
    
    summary = summarize_results(results)
    
    print("\n=== Scan Summary ===")
    print(f"Total files scanned: {summary['total_files']}")
    print(f"Supported files: {summary['supported_files']}")
    print(f"Unsupported files: {summary['unsupported_files']}")
    print(f"Total functions analyzed: {summary['total_functions']}")
    print(f"\nStatus distribution:")
    for status, count in summary["status_distribution"].items():
        print(f"  {status}: {count}")
    print(f"\nError distribution:")
    for error_type, count in summary["error_distribution"].items():
        print(f"  {error_type}: {count}")
    print(f"\nSuspicious cases found: {summary['suspicious_count']}")
    
    save_results(results, summary, args.output)
    
    if summary["suspicious_count"] > 0:
        print("\n=== Suspicious Cases ===")
        for case in summary["suspicious_cases"][:10]:
            print(f"File: {case['filepath']}")
            print(f"  Function: {case['function']}")
            for error in case["errors"]:
                print(f"    - {error['type']}: {error['message']}")


if __name__ == "__main__":
    main()