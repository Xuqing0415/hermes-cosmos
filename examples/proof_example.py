"""
Example: Neural-Symbolic Proof Generation

Demonstrates how to use the proof generation system.
"""

from hermes.neural_symbolic.proof_generator import ProofGenerator
from hermes.neural_symbolic.theorem_prover import TheoremProver
from hermes.neural_symbolic.proof_obligation import ProofObligationGenerator


def example_divide_function():
    """
    Example: Prove division safety
    
    Function: def divide(a, b): return a / b
    
    The system should:
    1. Detect division operation
    2. Generate verification condition: (not (= b 0))
    3. Attempt to prove b != 0
    4. Disprove and generate test case with b=0
    """
    code = """
def divide(a, b):
    return a / b
"""
    
    generator = ProofGenerator()
    result = generator.prove_and_generate_test(code, "divide")
    
    print("=" * 60)
    print("Example: Division Safety Proof")
    print("=" * 60)
    print(f"  Proven:   {result['proven']}")
    print(f"  Disproven: {result['disproven']}")
    print(f"  Unknown:  {result['unknown']}")
    print(f"  Duration: {result['duration']:.2f}s")
    
    if result["tests"]:
        print("\nGenerated Tests:")
        for test in result["tests"]:
            print(f"\n  Inputs:   {test['inputs']}")
            print(f"  Assertion: {test['assertion']}")
        
        print("\nGenerated Test Code:")
        print("-" * 40)
        print(result["test_code"])


def example_safe_divide_function():
    """
    Example: Prove safe division (with check)
    
    Function: def safe_divide(a, b):
                  if b == 0:
                      raise ValueError("Cannot divide by zero")
                  return a / b
    
    The system should:
    1. Detect division operation with guard
    2. Generate verification conditions
    3. Prove that division only happens when b != 0
    """
    code = """
def safe_divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    
    generator = ProofGenerator()
    result = generator.prove_and_generate_test(code, "safe_divide")
    
    print("\n" + "=" * 60)
    print("Example: Safe Division Proof")
    print("=" * 60)
    print(f"  Proven:   {result['proven']}")
    print(f"  Disproven: {result['disproven']}")
    print(f"  Unknown:  {result['unknown']}")
    print(f"  Duration: {result['duration']:.2f}s")
    
    if result["tests"]:
        print("\nGenerated Tests:")
        for test in result["tests"]:
            print(f"\n  Inputs:   {test['inputs']}")
            print(f"  Assertion: {test['assertion']}")
    else:
        print("\n No tests needed - all paths proven safe!")


def example_direct_proof():
    """
    Example: Direct theorem proving with Z3
    """
    prover = TheoremProver()
    
    print("\n" + "=" * 60)
    print("Example: Direct Theorem Proving")
    print("=" * 60)
    
    result1 = prover.verify_no_div_by_zero("b")
    print(f"\n1. Prove (not (= b 0)): {result1.status.value}")
    if result1.model:
        print(f"   Counterexample: {result1.model}")
    
    result2 = prover.prove("(assert (not (= x 0)))")
    print(f"\n2. Prove x != 0: {result2.status.value}")
    
    result3 = prover.prove_implication("(>= x 1)", "(> x 0)")
    print(f"\n3. Prove x >= 1 => x > 0: {result3.status.value}")


def example_obligation_generation():
    """
    Example: Generate proof obligations from code
    """
    code = """
def calculate(a, b):
    if a > b:
        return a / b
    else:
        return b - a
"""
    
    generator = ProofObligationGenerator()
    vcs = generator.generate_from_code(code, "calculate")
    
    print("\n" + "=" * 60)
    print("Example: Proof Obligation Generation")
    print("=" * 60)
    
    for i, vc in enumerate(vcs, 1):
        print(f"\nVC {i}: {vc.id}")
        print(f"  Description: {vc.description}")
        print(f"  SMT Formula: {vc.smt_formula}")
        print(f"  Source: {vc.source}")


if __name__ == "__main__":
    example_divide_function()
    example_safe_divide_function()
    example_direct_proof()
    example_obligation_generation()
