"""
SuperNet for Federated Neural Architecture Search

Implements a simplified search space and mixed operations for neural architecture search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class MixedOp(nn.Module):
    """
    Mixed operation that combines multiple candidate operations.

    Uses softmax over operation weights to select the best operation.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 op_names: Optional[List[str]] = None):
        super().__init__()

        self.ops = nn.ModuleDict({
            'sep_conv_3x3': self._make_sep_conv(in_channels, out_channels, 3),
            'sep_conv_5x5': self._make_sep_conv(in_channels, out_channels, 5),
            'skip_connect': self._make_skip_connect(in_channels, out_channels),
        })

        if op_names is None:
            op_names = list(self.ops.keys())
        self.op_names = op_names

        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1e-3)

    def _make_sep_conv(self, in_channels: int, out_channels: int, kernel_size: int):
        """Separable convolution."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                     padding=kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def _make_skip_connect(self, in_channels: int, out_channels: int):
        """Skip connection."""
        if in_channels == out_channels:
            return nn.Identity()
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, discrete: bool = False) -> torch.Tensor:
        """
        Forward pass through the mixed operation.

        Args:
            x: Input tensor
            discrete: If True, use hard selection (argmax)

        Returns:
            Output tensor
        """
        weights = F.softmax(self.alpha, dim=0)

        if discrete:
            best_op_idx = int(torch.argmax(weights).item())
            selected_op = list(self.ops.values())[best_op_idx]
        else:
            selected_op = list(self.ops.values())[int(torch.argmax(weights).item())]

        return selected_op(x)

    def get_best_op(self) -> nn.Module:
        """Get the operation with highest weight."""
        best_idx = int(torch.argmax(self.alpha).item())
        return list(self.ops.values())[best_idx]

    def get_alpha(self) -> torch.Tensor:
        """Get architecture weights."""
        return self.alpha.detach().clone()

    def set_alpha(self, alpha: torch.Tensor):
        """Set architecture weights."""
        self.alpha.data = alpha


class SimpleSuperNet(nn.Module):
    """
    Simplified Super network for federated NAS.

    A straightforward implementation with shared mixed operations.
    """

    def __init__(self, num_classes: int = 10,
                 num_cells: int = 6,
                 channels: int = 32):
        super().__init__()

        self.num_classes = num_classes
        self.num_cells = num_cells
        self.channels = channels

        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.cells = nn.ModuleList()
        for i in range(num_cells):
            cell = nn.ModuleDict({
                'conv1': MixedOp(channels, channels),
                'conv2': MixedOp(channels, channels),
            })
            self.cells.append(cell)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor, discrete: bool = False) -> torch.Tensor:
        """Forward pass through the super network."""
        x = F.relu(self.stem(x))

        for cell in self.cells:
            x1 = cell['conv1'](x, discrete=discrete)
            x2 = cell['conv2'](x1, discrete=discrete)
            x = x2 + x

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_alphas(self) -> List[torch.Tensor]:
        """Get all architecture weights."""
        alphas = []
        for cell in self.cells:
            alphas.append(cell['conv1'].get_alpha())
            alphas.append(cell['conv2'].get_alpha())
        return alphas

    def set_alphas(self, alphas: List[torch.Tensor]):
        """Set all architecture weights."""
        idx = 0
        for cell in self.cells:
            cell['conv1'].set_alpha(alphas[idx])
            idx += 1
            cell['conv2'].set_alpha(alphas[idx])
            idx += 1

    def get_arch_params(self) -> List[nn.Parameter]:
        """Get architecture parameters (alphas)."""
        params = []
        for cell in self.cells:
            params.append(cell['conv1'].alpha)
            params.append(cell['conv2'].alpha)
        return params

    def get_weight_params(self) -> List[nn.Parameter]:
        """Get weight parameters (excluding alphas)."""
        arch_params = set(self.get_arch_params())
        return [p for p in self.parameters() if p not in arch_params]


class SimpleSubNet(nn.Module):
    """
    Subnetwork extracted from the super network.
    """

    def __init__(self, supernet: SimpleSuperNet):
        super().__init__()
        self.supernet = supernet

        self.stem = supernet.stem
        self.cells = supernet.cells
        self.pool = supernet.pool
        self.classifier = supernet.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the subnet."""
        x = F.relu(self.stem(x))

        for cell in self.cells:
            x1 = cell['conv1'].get_best_op()(x)
            x2 = cell['conv2'].get_best_op()(x1)
            x = x2 + x

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 32, 32)) -> float:
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


class SearchSpace:
    """Base class for defining search spaces."""

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
    """

    def __init__(self):
        super().__init__()
        self.op_names = [
            'sep_conv_3x3',
            'sep_conv_5x5',
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
            self._skip_connect(in_channels, out_channels)
        ]

    def get_num_ops(self) -> int:
        """Get number of candidate operations."""
        return len(self.op_names)


SuperNet = SimpleSuperNet
SubNet = SimpleSubNet