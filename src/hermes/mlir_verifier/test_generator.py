from typing import Dict, List, Optional, Any
import re
import random
import time

from .types import TensorInput, KernelTest, VerificationResult


class MLIRTestGenerator:
    def __init__(self):
        self.shape_templates = [
            [4], [8], [16], [32], [64],
            [4, 4], [8, 8], [16, 16], [32, 32],
            [4, 4, 4], [8, 8, 8], [16, 16, 16],
            [1, 28, 28], [1, 32, 32], [3, 224, 224]
        ]
        self.dtypes = ["f32", "i32", "i64"]
    
    def generate_test_inputs(self, mlir_code: str, num_tests: int = 5) -> List[KernelTest]:
        memref_types = self._extract_memref_types(mlir_code)
        scalar_types = self._extract_scalar_types(mlir_code)
        
        tests = []
        for _ in range(num_tests):
            inputs = []
            for name, memref_type in memref_types.items():
                shape = self._generate_shape(memref_type["shape"])
                dtype = memref_type["dtype"]
                
                tensor_input = TensorInput(
                    name=name,
                    shape=shape,
                    dtype=dtype
                )
                tensor_input.generate_random_data()
                inputs.append(tensor_input)
            
            for name, scalar_type in scalar_types.items():
                tensor_input = TensorInput(
                    name=name,
                    shape=[],
                    dtype=scalar_type
                )
                tensor_input.generate_random_data()
                inputs.append(tensor_input)
            
            test = KernelTest(input_data=inputs)
            tests.append(test)
        
        return tests
    
    def _extract_scalar_types(self, mlir_code: str) -> Dict[str, str]:
        types = {}
        lines = mlir_code.strip().split('\n')
        
        for line in lines:
            if line.startswith('func.func'):
                match = re.match(r'func\.func @(\w+)\((.*?)\)', line)
                if match:
                    params = match.group(2)
                    param_list = params.split(',')
                    
                    for param in param_list:
                        param = param.strip()
                        if param:
                            parts = param.split(':')
                            if len(parts) == 2:
                                name = parts[0].strip().replace('%', '')
                                type_str = parts[1].strip()
                                
                                if type_str.startswith('i') or type_str.startswith('f'):
                                    types[name] = type_str
        
        return types
    
    def _extract_memref_types(self, mlir_code: str) -> Dict[str, Any]:
        types = {}
        lines = mlir_code.strip().split('\n')
        
        for line in lines:
            if line.startswith('func.func'):
                match = re.match(r'func\.func @(\w+)\((.*?)\)', line)
                if match:
                    params = match.group(2)
                    param_list = params.split(',')
                    
                    for param in param_list:
                        param = param.strip()
                        if param:
                            parts = param.split(':')
                            if len(parts) == 2:
                                name = parts[0].strip().replace('%', '')
                                type_str = parts[1].strip()
                                
                                if type_str.startswith('memref'):
                                    shape = self._parse_shape_from_memref(type_str)
                                    dtype = self._parse_dtype_from_memref(type_str)
                                    types[name] = {"shape": shape, "dtype": dtype}
        
        return types
    
    def _parse_shape_from_memref(self, type_str: str) -> List[int]:
        match = re.match(r'memref<(\[.*?\])x(\w+)>', type_str)
        if match:
            shape_str = match.group(1)
            dims = shape_str.strip('[]').split('x')
            shape = []
            for dim in dims:
                dim = dim.strip()
                if dim.isdigit():
                    shape.append(int(dim))
                else:
                    shape.append(-1)
            return shape
        
        return []
    
    def _parse_dtype_from_memref(self, type_str: str) -> str:
        match = re.match(r'memref<(\[.*?\])x(\w+)>', type_str)
        if match:
            return match.group(2)
        return "f32"
    
    def _generate_shape(self, template_shape: List[int]) -> List[int]:
        if all(dim > 0 for dim in template_shape):
            return template_shape
        
        shape = []
        for dim in template_shape:
            if dim > 0:
                shape.append(dim)
            else:
                shape.append(random.choice([4, 8, 16, 32]))
        
        return shape
    
    def generate_mlir_kernel(self, operation: str, shape: List[int], dtype: str = "f32") -> str:
        dim_str = "x".join(str(d) for d in shape)
        memref_type = f"memref<[{dim_str}]x{dtype}>"
        
        kernels = {
            "add": f"""func.func @add(%arg0: {memref_type}, %arg1: {memref_type}, %arg2: {memref_type}) {{
  %c0 = arith.constant 0 : index
  %c{shape[0]} = arith.constant {shape[0]} : index
  scf.for %i = %c0 to %c{shape[0]} step %c1 {{
    %val0 = memref.load %arg0[%i] : {memref_type}
    %val1 = memref.load %arg1[%i] : {memref_type}
    %result = arith.addf %val0, %val1 : {dtype}
    memref.store %result, %arg2[%i] : {memref_type}
  }}
  return
}}""",
            "mul": f"""func.func @mul(%arg0: {memref_type}, %arg1: {memref_type}, %arg2: {memref_type}) {{
  %c0 = arith.constant 0 : index
  %c{shape[0]} = arith.constant {shape[0]} : index
  scf.for %i = %c0 to %c{shape[0]} step %c1 {{
    %val0 = memref.load %arg0[%i] : {memref_type}
    %val1 = memref.load %arg1[%i] : {memref_type}
    %result = arith.mulf %val0, %val1 : {dtype}
    memref.store %result, %arg2[%i] : {memref_type}
  }}
  return
}}""",
            "transpose": f"""func.func @transpose(%arg0: {memref_type}, %arg1: {memref_type}) {{
  %c0 = arith.constant 0 : index
  %c{shape[0]} = arith.constant {shape[0]} : index
  scf.for %i = %c0 to %c{shape[0]} step %c1 {{
    %val = memref.load %arg0[%i] : {memref_type}
    memref.store %val, %arg1[%i] : {memref_type}
  }}
  return
}}"""
        }
        
        return kernels.get(operation, kernels["add"])
    
    def run_test(self, mlir_code: str, test: KernelTest) -> KernelTest:
        start_time = time.time()
        
        try:
            from .backend import get_mlir_backend
            
            backend = get_mlir_backend()
            
            input_data = {
                "inputs": [
                    {
                        "name": inpt.name,
                        "shape": inpt.shape,
                        "dtype": inpt.dtype,
                        "data": inpt.data
                    }
                    for inpt in test.input_data
                ]
            }
            
            result = backend.execute(mlir_code, input_data)
            test.passed = result.get("success", True)
            
            if "outputs" in result:
                test.actual_output = [
                    {
                        "name": out.get("name", ""),
                        "shape": out.get("shape", []),
                        "data": out.get("data", [])
                    }
                    for out in result["outputs"]
                ]
        
        except Exception as e:
            test.passed = False
        
        test.duration = time.time() - start_time
        return test
    
    def run_scalar_test(self, mlir_code: str, input_values: Dict[str, Any]) -> Dict[str, Any]:
        from .backend import get_mlir_backend
        
        backend = get_mlir_backend()
        
        inputs = []
        for name, value in input_values.items():
            dtype = "i32" if isinstance(value, int) else "f32"
            inputs.append({
                "name": name,
                "shape": [],
                "dtype": dtype,
                "data": [value]
            })
        
        result = backend.execute(mlir_code, {"inputs": inputs})
        
        if result.get("success"):
            expected = self._symbolic_compute(mlir_code, input_values)
            actual = result.get("result", None)
            
            if expected is not None and actual is not None:
                return {
                    "success": True,
                    "expected": expected,
                    "actual": actual,
                    "matched": expected == actual,
                    "backend": backend.get_name()
                }
        
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
            "backend": backend.get_name()
        }
    
    def _symbolic_compute(self, mlir_code: str, input_values: Dict[str, Any]) -> Optional[Any]:
        match = re.search(r'func\.func @(\w+)\((.*?)\)', mlir_code)
        if not match:
            return None
        
        func_name = match.group(1)
        
        if func_name == "add":
            if len(input_values) >= 2:
                vals = list(input_values.values())
                return vals[0] + vals[1]
        elif func_name == "mul":
            if len(input_values) >= 2:
                vals = list(input_values.values())
                return vals[0] * vals[1]
        elif func_name == "sub":
            if len(input_values) >= 2:
                vals = list(input_values.values())
                return vals[0] - vals[1]
        
        return None
    
    def generate_comprehensive_test_suite(self, mlir_code: str) -> Dict[str, Any]:
        tests = self.generate_test_inputs(mlir_code, num_tests=10)
        
        passed_count = 0
        for test in tests:
            result = self.run_test(mlir_code, test)
            if result.passed:
                passed_count += 1
        
        return {
            "total_tests": len(tests),
            "passed_tests": passed_count,
            "failed_tests": len(tests) - passed_count,
            "success_rate": passed_count / len(tests) if tests else 0,
            "tests": [{"shapes": [inpt.shape for inpt in t.input_data]} for t in tests]
        }