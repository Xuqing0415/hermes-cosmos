"""
SuperNet for Federated Neural Architecture Search

Implements the search space and mixed operations for neural architecture search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class MixedOp(nn.Module):
    """
    Mixed operation that combines multiple candidate operations.
    
    Uses softmax over operation weights to mix different operations.
    """
    
    def __init__(self, candidate_ops: List[nn.Module], op_names: Optional[List[str]] = None):
        super().__init__()
        self.candidate_ops = nn.ModuleList(candidate_ops)
        
        if op_names is None:
            op_names = [f'op_{i}' for i in range(len(candidate_ops))]
        self.op_names = op_names
        
        self.alpha = nn.Parameter(torch.randn(len(candidate_ops)) * 1e-3)
    
    def forward(self, x: torch.Tensor, discrete: bool = False) -> torch.Tensor:
        """
        Forward pass through the mixed operation.
        
        Args:
            x: Input tensor
            discrete: If True, use hard selection (argmax), otherwise use softmax
        
        Returns:
            Output tensor
        """
        if discrete:
            best_op_idx = int(torch.argmax(self.alpha).item())
            return self.candidate_ops[best_op_idx](x)
        else:
            weights = F.softmax(self.alpha, dim=0)
            output = sum(w * op(x) for w, op in zip(weights, self.candidate_ops))
            return output
    
    def get_best_op(self) -> nn.Module:
        """Get the operation with highest weight."""
        best_idx = int(torch.argmax(self.alpha).item())
        return self.candidate_ops[best_idx]
    
    def get_alpha(self) -> torch.Tensor:
        """Get architecture weights."""
        return self.alpha.detach().clone()
    
    def set_alpha(self, alpha: torch.Tensor):
        """Set architecture weights."""
        self.alpha.data = alpha
    
    def __repr__(self):
        return f"MixedOp(ops={self.op_names}, alpha={self.alpha.data})"


class SearchSpace:
    """
    Base class for defining search spaces.
    """
    
    def __init__(self):
        self.candidate_ops = []
        self.op_names = []
    
    def get_candidate_ops(self, in_channels: int, out_channels: int) -> List[nn.Module]:
        """Get candidate operations for a given layer."""
        raise NotImplementedError
    
    def get_num_ops(self) -> int:
        """Get number of candidate operations."""
        return len(self.candidate_ops)


class DartsSearchSpace(SearchSpace):
    """
    DARTS-style search space with common convolution operations.
    
    Candidate operations include:
    - 3x3 separable conv
    - 5x5 separable conv
    - 3x3 max pooling
    - 3x3 avg pooling
    - 3x3 dilated conv
    - 5x5 dilated conv
    - skip connection
    """
    
    def __init__(self):
        super().__init__()
        self.op_names = [
            'sep_conv_3x3',
            'sep_conv_5x5', 
            'max_pool_3x3',
            'avg_pool_3x3',
            'dil_conv_3x3',
            'dil_conv_5x5',
            'skip_connect'
        ]
    
    def _sep_conv(self, in_channels: int, out_channels: int, kernel_size: int):
        """Separable convolution."""
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def _dil_conv(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 2):
        """Dilated convolution."""
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
    
    def _skip_connect(self, in_channels: int, out_channels: int):
        """Skip connection."""
        if in_channels == out_channels:
            return nn.Identity()
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def get_candidate_ops(self, in_channels: int, out_channels: int) -> List[nn.Module]:
        """Get candidate operations."""
        return [
            self._sep_conv(in_channels, out_channels, 3),
            self._sep_conv(in_channels, out_channels, 5),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.AvgPool2d(3, stride=1, padding=1),
            self._dil_conv(in_channels, out_channels, 3, dilation=2),
            self._dil_conv(in_channels, out_channels, 5, dilation=2),
            self._skip_connect(in_channels, out_channels)
        ]
    
    def get_num_ops(self) -> int:
        """Get number of candidate operations."""
        return len(self.op_names)


class SuperNet(nn.Module):
    """
    Super network containing all possible operations.
    
    Each cell consists of multiple nodes connected by mixed operations.
    """
    
    def __init__(self, search_space: SearchSpace,
                 num_classes: int = 10,
                 num_cells: int = 8,
                 num_nodes: int = 4,
                 channels: int = 36):
        super().__init__()
        
        self.search_space = search_space
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.channels = channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
        self.cells = nn.ModuleList()
        self.reduction_indices = [num_cells // 3, 2 * num_cells // 3]
        
        c = channels
        for i in range(num_cells):
            if i in self.reduction_indices:
                c *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(search_space, c, c, reduction, num_nodes)
            self.cells.append(cell)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)
    
    def forward(self, x: torch.Tensor, discrete: bool = False) -> torch.Tensor:
        """
        Forward pass through the super network.
        
        Args:
            x: Input tensor
            discrete: If True, use discrete operation selection
        
        Returns:
            Output logits
        """
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell(x, discrete=discrete)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_alphas(self) -> List[torch.Tensor]:
        """Get all architecture weights."""
        alphas = []
        for cell in self.cells:
            for node in cell.nodes:
                for op in node:
                    alphas.append(op.get_alpha())
        return alphas
    
    def set_alphas(self, alphas: List[torch.Tensor]):
        """Set all architecture weights."""
        idx = 0
        for cell in self.cells:
            for node in cell.nodes:
                for op in node:
                    op.set_alpha(alphas[idx])
                    idx += 1
    
    def get_arch_params(self) -> List[nn.Parameter]:
        """Get architecture parameters (alphas)."""
        params = []
        for cell in self.cells:
            for node in cell.nodes:
                for op in node:
                    params.append(op.alpha)
        return params
    
    def get_weight_params(self) -> List[nn.Parameter]:
        """Get weight parameters (excluding alphas)."""
        arch_params = set(self.get_arch_params())
        return [p for p in self.parameters() if p not in arch_params]
    
    def extract_subnet(self, arch_weights: List[torch.Tensor]) -> nn.Module:
        """Extract a subnet from the super network."""
        subnet = SubNet(self, arch_weights)
        return subnet


class Cell(nn.Module):
    """
    Cell in the super network.
    
    Each cell contains multiple nodes with mixed operations between them.
    """
    
    def __init__(self, search_space: SearchSpace, 
                 in_channels: int, out_channels: int,
                 reduction: bool = False, num_nodes: int = 4):
        super().__init__()
        
        self.reduction = reduction
        self.num_nodes = num_nodes
        
        if reduction:
            self.stride = 2
        else:
            self.stride = 1
        
        self.preprocess0 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels // 2 if reduction else in_channels,
                     out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.preprocess1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.nodes = []
        
        for i in range(num_nodes):
            node_ops = []
            for j in range(i + 2):
                stride = self.stride if j < 2 and self.reduction else 1
                ops = search_space.get_candidate_ops(out_channels, out_channels)
                
                if stride == 2:
                    strided_ops = []
                    for op in ops:
                        strided_op = nn.Sequential(op, nn.MaxPool2d(2, stride=2))
                        strided_ops.append(strided_op)
                    mixed_op = MixedOp(strided_ops, search_space.op_names)
                else:
                    mixed_op = MixedOp(ops, search_space.op_names)
                
                node_ops.append(mixed_op)
            
            self.nodes.append(nn.ModuleList(node_ops))
        
        self.nodes = nn.ModuleList(self.nodes)
        
        self.concat = nn.Conv2d(num_nodes * out_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, discrete: bool = False) -> torch.Tensor:
        """Forward pass through the cell."""
        s0 = self.preprocess0(x)
        s1 = self.preprocess1(x)
        
        states = [s0, s1]
        
        for i, node_ops in enumerate(self.nodes):
            outputs = []
            for j, op in enumerate(node_ops):
                outputs.append(op(states[j], discrete=discrete))
            
            node_output = sum(outputs)
            states.append(node_output)
        
        concat_out = torch.cat(states[2:], dim=1)
        return self.concat(concat_out)


class SubNet(nn.Module):
    """
    Subnetwork extracted from the super network.
    
    Contains only the selected operations based on architecture weights.
    """
    
    def __init__(self, supernet: SuperNet, arch_weights: List[torch.Tensor]):
        super().__init__()
        
        self.supernet = supernet
        self.arch_weights = arch_weights
        
        self.subnet = self._build_subnet()
    
    def _build_subnet(self) -> nn.Module:
        """Build the subnet from architecture weights."""
        subnet = nn.Sequential()
        
        subnet.add_module('stem', self.supernet.stem)
        
        alpha_idx = 0
        for cell_idx, cell in enumerate(self.supernet.cells):
            subnet_cell = self._extract_cell(cell, alpha_idx)
            alpha_idx += sum(len(node) for node in cell.nodes)
            subnet.add_module(f'cell_{cell_idx}', subnet_cell)
        
        subnet.add_module('global_pooling', self.supernet.global_pooling)
        subnet.add_module('classifier', self.supernet.classifier)
        
        return subnet
    
    def _extract_cell(self, cell: Cell, alpha_idx: int) -> nn.Module:
        """Extract a cell from the super network."""
        subnet_cell = Cell(self.supernet.search_space,
                          cell.preprocess0[1].in_channels,
                          cell.preprocess0[1].out_channels,
                          cell.reduction,
                          cell.num_nodes)
        
        subnet_cell.preprocess0 = cell.preprocess0
        subnet_cell.preprocess1 = cell.preprocess1
        
        for i, node in enumerate(cell.nodes):
            for j, op in enumerate(node):
                best_op = op.get_best_op()
                subnet_cell.nodes[i][j] = best_op
        
        subnet_cell.concat = cell.concat
        
        return subnet_cell
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the subnet."""
        return self.subnet(x)
    
    def get_flops(self, input_size: Tuple[int, int, int] = (3, 32, 32)) -> float:
        """Estimate FLOPs for the subnet."""
        return estimate_flops(self.subnet, input_size)


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int]) -> float:
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
    
    dummy_input = torch.randn(1, *input_size)
    model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return flops