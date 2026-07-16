"""
CLI for Neural-Symbolic Proof Generation
"""

import argparse
import sys
import os
from typing import Optional

from hermes.neural_symbolic.proof_generator import ProofGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Neural-Symbolic Proof Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    prove_parser = subparsers.add_parser("prove", help="Prove a function")
    prove_parser.add_argument("--function", required=True, help="Function name to prove")
    prove_parser.add_argument("--module", required=True, help="Python module file")
    prove_parser.add_argument("--solver", default="z3", help="Theorem prover solver")
    prove_parser.add_argument("--timeout", type=int, default=30, help="Proof timeout in seconds")
    prove_parser.add_argument("--output", help="Output file for results")
    
    verify_parser = subparsers.add_parser("verify", help="Verify division safety")
    verify_parser.add_argument("--function", required=True, help="Function name to verify")
    verify_parser.add_argument("--module", required=True, help="Python module file")
    
    test_parser = subparsers.add_parser("test", help="Generate tests from proof")
    test_parser.add_argument("--function", required=True, help="Function name")
    test_parser.add_argument("--module", required=True, help="Python module file")
    test_parser.add_argument("--output", default="generated_tests.py", help="Output test file")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(args.module):
        print(f"Error: Module file '{args.module}' not found")
        sys.exit(1)
    
    with open(args.module, "r") as f:
        code = f.read()
    
    generator = ProofGenerator(solver=args.solver, timeout=args.timeout)
    
    if args.command == "prove":
        result = generator.prove_and_generate_test(code, args.function)
        print_results(result)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(str(result))
            print(f"\nResults saved to {args.output}")
    
    elif args.command == "verify":
        result = generator.verify_division_safety(code, args.function)
        print_verification(result)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(str(result))
            print(f"\nResults saved to {args.output}")
    
    elif args.command == "test":
        result = generator.prove_and_generate_test(code, args.function)
        
        if result["test_code"]:
            with open(args.output, "w") as f:
                f.write(result["test_code"])
            print(f"Test file generated: {args.output}")
            print(f"Generated {len(result['tests'])} test(s)")
        else:
            print("No tests generated - all proofs succeeded!")


def print_results(result):
    """Print proof results in a readable format"""
    print("=" * 60)
    print("Neural-Symbolic Proof Results")
    print("=" * 60)
    print(f"  Proven:   {result['proven']}")
    print(f"  Disproven: {result['disproven']}")
    print(f"  Unknown:  {result['unknown']}")
    print(f"  Duration: {result['duration']:.2f}s")
    print()
    
    if result["tests"]:
        print("Generated Tests:")
        print("-" * 40)
        for i, test in enumerate(result["tests"], 1):
            print(f"\nTest {i}: {test['id']}")
            print(f"  Function: {test['function_name']}")
            print(f"  Inputs:   {test['inputs']}")
            print(f"  Assertion: {test['assertion']}")
            print(f"  Description: {test['description']}")
        
        print("\nTest Code:")
        print("-" * 40)
        print(result["test_code"])


def print_verification(result):
    """Print verification results in a readable format"""
    print("=" * 60)
    print("Division Safety Verification")
    print("=" * 60)
    print(f"  Result: {result['result']}")
    print(f"  Safe:   {result['safe']}")
    print(f"  Proof Results: {result['proof_results']}")
    print(f"  Generated Tests: {result['generated_tests']}")
    
    if result["test_code"]:
        print("\nGenerated Test Code:")
        print("-" * 40)
        print(result["test_code"])


if __name__ == "__main__":
    main()
