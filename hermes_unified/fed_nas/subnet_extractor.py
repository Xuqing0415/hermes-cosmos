"""
Subnetwork Extractor for Federated NAS

Implements resource evaluation and constraint checking for subnet extraction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class ResourceEvaluator:
    """
    Evaluates resource usage of neural networks.
    
    Measures FLOPs, latency, memory usage, and parameter count.
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
    
    def evaluate_flops(self, model: nn.Module, 
                      input_size: Tuple[int, int, int] = (3, 32, 32)) -> float:
        """
        Estimate FLOPs for a model.
        
        Args:
            model: Model to evaluate
            input_size: Input size (channels, height, width)
        
        Returns:
            Estimated FLOPs
        """
        flops = 0
        
        def count_flops(module, input, output):
            nonlocal flops
            
            if isinstance(module, nn.Conv2d):
                cin = module.in_channels
                cout = module.out_channels
                kh, kw = module.kernel_size
                batch_size = input[0].size(0)
                out_h, out_w = output.size()[2:]
                
                flops += batch_size * cin * cout * kh * kw * out_h * out_w
            
            elif isinstance(module, nn.Linear):
                cin = module.in_features
                cout = module.out_features
                batch_size = input[0].size(0)
                
                flops += batch_size * cin * cout
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(count_flops))
        
        dummy_input = torch.randn(1, *input_size).to(self.device)
        model.to(self.device)
        model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def evaluate_latency(self, model: nn.Module,
                        input_size: Tuple[int, int, int] = (3, 32, 32),
                        warmup: int = 5, iterations: int = 100) -> float:
        """
        Measure inference latency.
        
        Args:
            model: Model to evaluate
            input_size: Input size
            warmup: Number of warmup iterations
            iterations: Number of measurement iterations
        
        Returns:
            Average latency in milliseconds
        """
        model.to(self.device)
        model.eval()
        
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        for _ in range(warmup):
            model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if self.device.type == 'cuda':
            start_time.record()
        
        total_time = 0.0
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            
            if self.device.type == 'cuda':
                start.record()
                model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
            else:
                import time
                start = time.time()
                model(dummy_input)
                end = time.time()
                total_time += (end - start) * 1000
        
        if self.device.type == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
        
        return total_time / iterations
    
    def evaluate_memory(self, model: nn.Module) -> float:
        """
        Estimate memory usage.
        
        Args:
            model: Model to evaluate
        
        Returns:
            Memory usage in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 ** 2)
        return total_size
    
    def evaluate_params(self, model: nn.Module) -> int:
        """
        Count number of parameters.
        
        Args:
            model: Model to evaluate
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in model.parameters())
    
    def evaluate(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 32, 32)) -> Dict[str, Any]:
        """
        Perform comprehensive resource evaluation.
        
        Args:
            model: Model to evaluate
            input_size: Input size
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            'flops': self.evaluate_flops(model, input_size),
            'latency': self.evaluate_latency(model, input_size),
            'memory': self.evaluate_memory(model),
            'params': self.evaluate_params(model)
        }


class ConstraintChecker:
    """
    Checks if a model satisfies resource constraints.
    """
    
    def __init__(self, max_flops: Optional[float] = None,
                 max_latency: Optional[float] = None,
                 max_memory: Optional[float] = None,
                 max_params: Optional[int] = None):
        self.max_flops = max_flops
        self.max_latency = max_latency
        self.max_memory = max_memory
        self.max_params = max_params
    
    def check(self, resources: Dict[str, Any]) -> bool:
        """
        Check if resources satisfy all constraints.
        
        Args:
            resources: Dictionary of resource metrics
        
        Returns:
            True if all constraints are satisfied
        """
        if self.max_flops is not None and resources.get('flops', 0) > self.max_flops:
            return False
        
        if self.max_latency is not None and resources.get('latency', 0) > self.max_latency:
            return False
        
        if self.max_memory is not None and resources.get('memory', 0) > self.max_memory:
            return False
        
        if self.max_params is not None and resources.get('params', 0) > self.max_params:
            return False
        
        return True
    
    def get_violations(self, resources: Dict[str, Any]) -> List[str]:
        """
        Get list of violated constraints.
        
        Args:
            resources: Dictionary of resource metrics
        
        Returns:
            List of violated constraint names
        """
        violations = []
        
        if self.max_flops is not None and resources.get('flops', 0) > self.max_flops:
            violations.append('flops')
        
        if self.max_latency is not None and resources.get('latency', 0) > self.max_latency:
            violations.append('latency')
        
        if self.max_memory is not None and resources.get('memory', 0) > self.max_memory:
            violations.append('memory')
        
        if self.max_params is not None and resources.get('params', 0) > self.max_params:
            violations.append('params')
        
        return violations


class SubnetExtractor:
    """
    Extracts subnets from a super network based on architecture weights.
    """
    
    def __init__(self, supernet: nn.Module, resource_evaluator: ResourceEvaluator):
        self.supernet = supernet
        self.resource_evaluator = resource_evaluator
    
    def extract_subnet(self, arch_weights: List[torch.Tensor],
                      discrete: bool = True) -> nn.Module:
        """
        Extract a subnet from the super network.
        
        Args:
            arch_weights: List of architecture weights for each mixed operation
            discrete: If True, use argmax selection
        
        Returns:
            Extracted subnet
        """
        subnet = self._build_subnet(arch_weights, discrete)
        return subnet
    
    def _build_subnet(self, arch_weights: List[torch.Tensor],
                     discrete: bool) -> nn.Module:
        """Build subnet from architecture weights."""
        subnet = nn.Sequential()
        
        subnet.add_module('stem', self.supernet.stem)
        
        alpha_idx = 0
        for cell_idx, cell in enumerate(self.supernet.cells):
            subnet_cell = self._extract_cell(cell, arch_weights, alpha_idx, discrete)
            alpha_idx += sum(len(node) for node in cell.nodes)
            subnet.add_module(f'cell_{cell_idx}', subnet_cell)
        
        subnet.add_module('global_pooling', self.supernet.global_pooling)
        subnet.add_module('classifier', self.supernet.classifier)
        
        return subnet
    
    def _extract_cell(self, cell, arch_weights: List[torch.Tensor],
                     alpha_idx: int, discrete: bool) -> nn.Module:
        """Extract a cell from the super network."""
        subnet_cell = type(cell)(self.supernet.search_space,
                                cell.preprocess0[1].in_channels,
                                cell.preprocess0[1].out_channels,
                                cell.reduction,
                                cell.num_nodes)
        
        subnet_cell.preprocess0 = cell.preprocess0
        subnet_cell.preprocess1 = cell.preprocess1
        
        for i, node in enumerate(cell.nodes):
            for j, op in enumerate(node):
                if discrete:
                    best_idx = int(torch.argmax(arch_weights[alpha_idx]).item())
                    best_op = op.candidate_ops[best_idx]
                    subnet_cell.nodes[i][j] = best_op
                else:
                    subnet_cell.nodes[i][j] = op
                
                alpha_idx += 1
        
        subnet_cell.concat = cell.concat
        
        return subnet_cell
    
    def extract_and_evaluate(self, arch_weights: List[torch.Tensor],
                            input_size: Tuple[int, int, int] = (3, 32, 32)) -> Dict[str, Any]:
        """
        Extract subnet and evaluate its resources.
        
        Args:
            arch_weights: List of architecture weights
            input_size: Input size for evaluation
        
        Returns:
            Dictionary containing subnet and resource metrics
        """
        subnet = self.extract_subnet(arch_weights)
        resources = self.resource_evaluator.evaluate(subnet, input_size)
        
        return {
            'subnet': subnet,
            'resources': resources,
            'arch_weights': arch_weights
        }


class ResourcePredictor:
    """
    Predicts resource usage from architecture description without building the model.
    """
    
    def __init__(self):
        self.flop_cache = {}
    
    def predict_flops(self, arch_description: Dict[str, Any],
                     input_size: Tuple[int, int, int] = (3, 32, 32)) -> float:
        """
        Predict FLOPs from architecture description.
        
        Args:
            arch_description: Architecture description
            input_size: Input size
        
        Returns:
            Predicted FLOPs
        """
        flops = 0
        
        for cell in arch_description.get('cells', []):
            for node in cell.get('nodes', []):
                for edge in node.get('edges', []):
                    op_type = edge.get('op_type')
                    in_channels = edge.get('in_channels', 36)
                    out_channels = edge.get('out_channels', 36)
                    
                    if op_type in ['sep_conv_3x3', 'sep_conv_5x5']:
                        kernel_size = 3 if '3x3' in op_type else 5
                        flops += 2 * in_channels * out_channels * kernel_size * kernel_size * input_size[1] * input_size[2]
                    
                    elif op_type in ['dil_conv_3x3', 'dil_conv_5x5']:
                        kernel_size = 3 if '3x3' in op_type else 5
                        flops += in_channels * out_channels * kernel_size * kernel_size * input_size[1] * input_size[2]
        
        return flops
    
    def predict_latency(self, arch_description: Dict[str, Any]) -> float:
        """
        Predict latency from architecture description.
        
        Args:
            arch_description: Architecture description
        
        Returns:
            Predicted latency in ms
        """
        base_latency = 1.0
        num_cells = len(arch_description.get('cells', []))
        
        latency = base_latency * num_cells
        
        return latency
    
    def predict(self, arch_description: Dict[str, Any],
               input_size: Tuple[int, int, int] = (3, 32, 32)) -> Dict[str, float]:
        """
        Predict all resource metrics.
        
        Args:
            arch_description: Architecture description
            input_size: Input size
        
        Returns:
            Dictionary of predicted metrics
        """
        return {
            'flops': self.predict_flops(arch_description, input_size),
            'latency': self.predict_latency(arch_description),
            'memory': self.predict_flops(arch_description, input_size) / 1e6,
            'params': self.predict_flops(arch_description, input_size) / 1000
        }