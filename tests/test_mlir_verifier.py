import pytest

from hermes.mlir_verifier import (
    MLIRSymbolicExecutor,
    MLIRTestGenerator,
    MLIRPassVerifier,
    CompilerVerifier,
    VerificationResult,
    ErrorType
)


class TestMLIRSymbolicExecutor:
    def test_parse_simple_function(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        cfg = executor.parse_mlir_to_cfg(mlir_code)
        assert "entry" in cfg
    
    def test_cfg_with_multiple_blocks(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi eq, %arg0, %c0 : i32
  cf.cond_br %cmp, ^then, ^else
^then:
  %result = arith.constant 1 : i32
  cf.br ^end
^else:
  %result = arith.constant 2 : i32
  cf.br ^end
^end:
  return %result : i32
}"""
        
        cfg = executor.parse_mlir_to_cfg(mlir_code)
        assert "entry" in cfg
        assert "then" in cfg
        assert "else" in cfg
        assert "end" in cfg
        
        entry_block = cfg["entry"]
        assert len(entry_block.successors) == 2
        
        then_block = cfg["then"]
        assert len(then_block.successors) == 1
        assert then_block.successors[0] == "end"
    
    def test_verify_valid_load(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[5] : memref<[10]xf32>
  return
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "passed"
        assert len(result.errors) == 0
    
    def test_verify_out_of_bounds_load(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[15] : memref<[10]xf32>
  return
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "failed"
        assert len(result.errors) == 1
        assert result.errors[0].type == ErrorType.INDEX_OUT_OF_BOUNDS
    
    def test_verify_out_of_bounds_store(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %c0 = arith.constant 0.0 : f32
  memref.store %c0, %arg0[20] : memref<[10]xf32>
  return
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "failed"
        assert len(result.errors) == 1
    
    def test_arithmetic_operations(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32, %b: i32) -> i32 {
  %add = arith.addi %a, %b : i32
  %mul = arith.muli %a, %b : i32
  return %add : i32
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "passed"
    
    def test_comparison_operations(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32, %b: i32) -> i1 {
  %cmp = arith.cmpi slt, %a, %b : i32
  return %cmp : i1
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "passed"
    
    def test_symbolic_state(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %add = arith.addi %a, %c5 : i32
  return %add : i32
}"""
        
        executor.execute(mlir_code)
        state = executor.get_symbolic_state()
        
        assert "a" in state["values"]
        assert "c5" in state["values"]
        assert "add" in state["values"]
    
    def test_get_cfg_info(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %cmp = arith.cmpi eq, %a, %c0 : i32
  cf.cond_br %cmp, ^then, ^else
^then:
  %result = arith.constant 1 : i32
  cf.br ^end
^else:
  %result = arith.constant 2 : i32
  cf.br ^end
^end:
  return %result : i32
}"""
        
        executor.execute(mlir_code)
        info = executor.get_cfg_info()
        
        assert "entry" in info
        assert "then" in info
        assert "else" in info
        assert "end" in info
    
    def test_generate_cfg_dot(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32) -> i32 {
  return %a : i32
}"""
        
        executor.parse_mlir_to_cfg(mlir_code)
        dot_content = executor.generate_cfg_dot()
        
        assert "digraph CFG" in dot_content
        assert "entry" in dot_content


class TestMLIRTestGenerator:
    def test_extract_memref_types(self):
        generator = MLIRTestGenerator()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>, %arg1: memref<[5]xi32>) {
  return
}"""
        
        types = generator._extract_memref_types(mlir_code)
        assert "arg0" in types
        assert types["arg0"]["dtype"] == "f32"
        assert types["arg0"]["shape"] == [10]
    
    def test_generate_test_inputs(self):
        generator = MLIRTestGenerator()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        tests = generator.generate_test_inputs(mlir_code, num_tests=3)
        assert len(tests) == 3
        assert len(tests[0].input_data) == 1
        assert tests[0].input_data[0].name == "arg0"
    
    def test_generate_random_data(self):
        from hermes.mlir_verifier.types import TensorInput
        
        tensor = TensorInput(name="test", shape=[4], dtype="f32")
        data = tensor.generate_random_data()
        
        assert len(data) == 4
        assert all(isinstance(d, float) for d in data)
    
    def test_generate_mlir_kernel(self):
        generator = MLIRTestGenerator()
        
        kernel = generator.generate_mlir_kernel("add", [8], "f32")
        assert "func.func @add" in kernel
        assert "arith.addf" in kernel
        assert "memref.load" in kernel
        assert "memref.store" in kernel
    
    def test_generate_comprehensive_test_suite(self):
        generator = MLIRTestGenerator()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        suite = generator.generate_comprehensive_test_suite(mlir_code)
        assert "total_tests" in suite
        assert "passed_tests" in suite
        assert suite["total_tests"] == 10


class TestMLIRPassVerifier:
    def test_apply_constant_fold(self):
        verifier = MLIRPassVerifier()
        mlir_code = """func.func @test() {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return
}"""
        
        result = verifier.apply_pass_and_verify(mlir_code, "constant_fold")
        assert result.is_equivalent
        assert result.passed_tests == result.num_tests
    
    def test_verify_equivalence(self):
        verifier = MLIRPassVerifier()
        original = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        optimized = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        result = verifier.verify_equivalence(original, optimized)
        assert result.is_equivalent
    
    def test_batch_verify_passes(self):
        verifier = MLIRPassVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        results = verifier.batch_verify_passes(mlir_code, ["canonicalize", "constant_fold"])
        assert len(results) == 2
        assert "canonicalize" in results
        assert "constant_fold" in results


class TestCompilerVerifier:
    def test_init(self):
        verifier = CompilerVerifier()
        assert verifier.role.value == "compiler_verifier"
        assert "mlir_verification" in verifier.capabilities
    
    def test_verify_mlir(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[5] : memref<[10]xf32>
  return
}"""
        
        result = verifier.execute_task("mlir_verification", {"mlir_code": mlir_code})
        assert result["status"] == "passed"
        assert result["errors"] == []
    
    def test_detect_bugs(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[15] : memref<[10]xf32>
  return
}"""
        
        result = verifier.execute_task("bug_detection", {"mlir_code": mlir_code})
        assert result["total_bugs"] >= 1
        assert len(result["bugs_found"]) >= 1
    
    def test_validate_pass(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        result = verifier.execute_task("pass_validation", {"mlir_code": mlir_code, "pass_name": "canonicalize"})
        assert "is_equivalent" in result
    
    def test_prove_equivalence(self):
        verifier = CompilerVerifier()
        original = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        optimized = """func.func @test(%arg0: memref<[10]xf32>) {
  return
}"""
        
        result = verifier.execute_task("equivalence_proof", {
            "original_code": original,
            "optimized_code": optimized
        })
        assert result["is_equivalent"]
        assert result["confidence"] == 1.0
    
    def test_function_verification(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}"""
        
        test_cases = [
            {"a": 1, "b": 2},
            {"a": -10, "b": 5},
            {"a": 100, "b": 200}
        ]
        
        result = verifier.execute_task("function_verification", {
            "mlir_code": mlir_code,
            "test_cases": test_cases
        })
        
        assert "symbolic_status" in result
        assert "test_results" in result
        assert len(result["test_results"]) == 3


class TestMLIRSymbolicExecutorAdvanced:
    def test_arithmetic_operation_parsing(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}"""
        
        cfg = executor.parse_mlir_to_cfg(mlir_code)
        entry_block = cfg["entry"]
        arith_instrs = [i for i in entry_block.instructions if i["type"] in ["addi", "subi", "muli"]]
        assert len(arith_instrs) >= 1
    
    def test_scalar_type_extraction(self):
        generator = MLIRTestGenerator()
        mlir_code = """func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}"""
        
        types = generator._extract_scalar_types(mlir_code)
        assert "a" in types
        assert types["a"] == "i32"
        assert "b" in types
        assert types["b"] == "i32"
    
    def test_generate_test_inputs_with_scalars(self):
        generator = MLIRTestGenerator()
        mlir_code = """func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}"""
        
        tests = generator.generate_test_inputs(mlir_code, num_tests=3)
        assert len(tests) == 3
        assert len(tests[0].input_data) == 2


class TestMLIRBackend:
    def test_backend_manager_init(self):
        from hermes.mlir_verifier import MLIRBackendManager
        
        manager = MLIRBackendManager.get_instance()
        assert manager is not None
        
        backends = manager.list_backends()
        assert len(backends) >= 1
        
        current_backend = manager.get_backend()
        assert current_backend.is_available()
    
    def test_mock_backend_execution(self):
        from hermes.mlir_verifier import MockMLIRBackend
        
        backend = MockMLIRBackend()
        assert backend.is_available()
        
        mlir_code = """func.func @test(%arg0: memref<[4]xf32>) {
  return
}"""
        
        result = backend.execute(mlir_code, {"inputs": [{"name": "arg0", "shape": [4], "data": [1.0, 2.0, 3.0, 4.0]}]})
        assert result["success"]
    
    def test_backend_auto_selection(self):
        from hermes.mlir_verifier import MLIRBackendManager, get_mlir_backend
        
        manager = MLIRBackendManager.get_instance()
        backend = get_mlir_backend()
        
        assert backend.get_name() in ["python", "subprocess", "mock"]
        assert backend.is_available()


class TestEndToEndVerification:
    def test_symbolic_execution_with_z3(self):
        executor = MLIRSymbolicExecutor()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[5] : memref<[10]xf32>
  return
}"""
        
        result = executor.execute(mlir_code)
        assert result.status == "passed"
        assert len(result.errors) == 0
    
    def test_full_verification_pipeline(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[5] : memref<[10]xf32>
  %c0 = arith.constant 0.0 : f32
  memref.store %c0, %arg0[3] : memref<[10]xf32>
  return
}"""
        
        result = verifier.execute_task("mlir_verification", {"mlir_code": mlir_code})
        assert result["status"] == "passed"
        assert "test_results" in result
    
    def test_bug_detection_pipeline(self):
        verifier = CompilerVerifier()
        mlir_code = """func.func @test(%arg0: memref<[10]xf32>) {
  %val = memref.load %arg0[15] : memref<[10]xf32>
  return
}"""
        
        result = verifier.execute_task("bug_detection", {"mlir_code": mlir_code})
        assert result["total_bugs"] >= 1
        assert "test_coverage" in result