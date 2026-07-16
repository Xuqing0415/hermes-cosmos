from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import subprocess
import os
import tempfile
import json


class MLIRBackend(ABC):
    @abstractmethod
    def execute(self, mlir_code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class MockMLIRBackend(MLIRBackend):
    def execute(self, mlir_code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = input_data.get("inputs", [])
        outputs = []
        
        for inp in inputs:
            output_shape = inp.get("shape", [])
            output_data = [0.0] * (1 if not output_shape else eval('*'.join(map(str, output_shape))))
            outputs.append({
                "name": inp.get("name", "") + "_out",
                "shape": output_shape,
                "data": output_data
            })
        
        return {
            "success": True,
            "outputs": outputs,
            "backend": self.get_name()
        }
    
    def is_available(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "mock"


class SubprocessMLIRBackend(MLIRBackend):
    def __init__(self, mlir_cpu_runner_path: Optional[str] = None):
        self._mlir_cpu_runner_path = mlir_cpu_runner_path or self._find_mlir_cpu_runner()
        self._available = self._check_availability()
    
    def _find_mlir_cpu_runner(self) -> Optional[str]:
        paths = [
            "mlir-cpu-runner",
            "/usr/local/bin/mlir-cpu-runner",
            "/opt/mlir/bin/mlir-cpu-runner",
            "C:\\mlir\\bin\\mlir-cpu-runner.exe"
        ]
        
        for path in paths:
            try:
                result = subprocess.run([path, "--help"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
                continue
        
        return None
    
    def _check_availability(self) -> bool:
        return self._mlir_cpu_runner_path is not None
    
    def execute(self, mlir_code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "MLIR backend not available"}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                f.write(mlir_code)
                mlir_file = f.name
            
            args = [self._mlir_cpu_runner_path]
            
            args.extend([
                "-e", "main",
                "-entry-point-result=void"
            ])
            
            result = subprocess.run(
                args,
                input=mlir_code.encode(),
                capture_output=True,
                timeout=30
            )
            
            os.unlink(mlir_file)
            
            if result.returncode == 0:
                outputs = self._parse_output(result.stdout.decode())
                return {
                    "success": True,
                    "outputs": outputs,
                    "backend": self.get_name(),
                    "stdout": result.stdout.decode(),
                    "stderr": result.stderr.decode()
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr.decode(),
                    "backend": self.get_name(),
                    "return_code": result.returncode
                }
        
        except Exception as e:
            return {"success": False, "error": str(e), "backend": self.get_name()}
    
    def _parse_output(self, output: str) -> List[Dict[str, Any]]:
        outputs = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if 'memref<' in line:
                match = self._parse_memref_output(line)
                if match:
                    outputs.append(match)
        
        return outputs
    
    def _parse_memref_output(self, line: str) -> Optional[Dict[str, Any]]:
        import re
        
        match = re.match(r'memref<(\[.*?\])x(\w+)> = (.*)', line)
        if match:
            shape_str = match.group(1)
            dtype = match.group(2)
            data_str = match.group(3)
            
            shape = []
            dims = shape_str.strip('[]').split('x')
            for dim in dims:
                dim = dim.strip()
                if dim.isdigit():
                    shape.append(int(dim))
            
            data = []
            try:
                data = json.loads(data_str.replace(' ', ','))
            except (json.JSONDecodeError, ValueError):
                pass
            
            return {
                "shape": shape,
                "dtype": dtype,
                "data": data
            }
        
        return None
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "subprocess"


class PythonMLIRBackend(MLIRBackend):
    def __init__(self):
        self._available = self._check_availability()
        self._context = None
        self._module = None
    
    def _check_availability(self) -> bool:
        try:
            import mlir.ir
            import mlir.passmanager
            return True
        except ImportError:
            return False
    
    def execute(self, mlir_code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "MLIR Python bindings not available"}
        
        try:
            import mlir.ir
            import mlir.passmanager
            import mlir.execution_engine
            
            with mlir.ir.Context() as ctx:
                ctx.allow_unregistered_dialects = True
                
                module = mlir.ir.Module.parse(mlir_code)
                
                pm = mlir.passmanager.PassManager.parse(
                    "builtin.module(convert-scf-to-cf, convert-arith-to-llvm, convert-func-to-llvm, "
                    "convert-memref-to-llvm, finalize-memref-to-llvm)"
                )
                pm.run(module)
                
                execution_engine = mlir.execution_engine.ExecutionEngine(module)
                
                memref_args = []
                for inp in input_data.get("inputs", []):
                    import ctypes
                    import numpy as np
                    
                    data = np.array(inp.get("data", []), dtype=np.float32 if inp.get("dtype") == "f32" else np.int32)
                    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    memref_args.append(data_ptr)
                
                func_name = "main"
                if "@" in mlir_code:
                    import re
                    match = re.search(r'func\.func @(\w+)', mlir_code)
                    if match:
                        func_name = match.group(1)
                
                result = execution_engine.invoke(f"_mlir_ciface_{func_name}", *memref_args)
                
                return {
                    "success": True,
                    "outputs": [],
                    "backend": self.get_name(),
                    "result": result
                }
        
        except Exception as e:
            return {"success": False, "error": str(e), "backend": self.get_name()}
    
    def parse_and_analyze(self, mlir_code: str) -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "MLIR Python bindings not available"}
        
        try:
            import mlir.ir
            
            with mlir.ir.Context() as ctx:
                ctx.allow_unregistered_dialects = True
                
                module = mlir.ir.Module.parse(mlir_code)
                
                operations = []
                for op in module.body.operations:
                    op_info = {
                        "name": op.name,
                        "attributes": {str(name): str(value) for name, value in op.attributes.items()},
                        "num_operands": len(op.operands),
                        "num_results": len(op.results),
                        "location": str(op.location)
                    }
                    operations.append(op_info)
                
                return {
                    "success": True,
                    "operations": operations,
                    "backend": self.get_name()
                }
        
        except Exception as e:
            return {"success": False, "error": str(e), "backend": self.get_name()}
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "python"


class MLIRBackendManager:
    _instance = None
    
    def __init__(self):
        self._backends: List[MLIRBackend] = [
            PythonMLIRBackend(),
            SubprocessMLIRBackend(),
            MockMLIRBackend()
        ]
        self._current_backend = self._select_backend()
    
    @classmethod
    def get_instance(cls) -> "MLIRBackendManager":
        if cls._instance is None:
            cls._instance = MLIRBackendManager()
        return cls._instance
    
    def _select_backend(self) -> MLIRBackend:
        for backend in self._backends:
            if backend.is_available():
                return backend
        return self._backends[-1]
    
    def get_backend(self) -> MLIRBackend:
        return self._current_backend
    
    def set_backend(self, backend_name: str) -> bool:
        for backend in self._backends:
            if backend.get_name() == backend_name and backend.is_available():
                self._current_backend = backend
                return True
        return False
    
    def list_backends(self) -> List[Dict[str, Any]]:
        return [
            {"name": b.get_name(), "available": b.is_available()}
            for b in self._backends
        ]


def get_mlir_backend() -> MLIRBackend:
    return MLIRBackendManager.get_instance().get_backend()