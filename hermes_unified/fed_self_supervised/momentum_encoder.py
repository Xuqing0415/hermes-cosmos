"""
Momentum Encoder for Federated Self-Supervised Learning

Implements federated momentum encoder with exponential moving average updates.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional


class MomentumEncoder(nn.Module):
    """Momentum encoder with exponential moving average updates."""
    
    def __init__(self, base_encoder: nn.Module, momentum: float = 0.999):
        """
        Initialize momentum encoder.
        
        Args:
            base_encoder: Base encoder network
            momentum: Momentum factor for EMA updates
        """
        super().__init__()
        
        self.momentum = momentum
        self.encoder = self._create_encoder(base_encoder)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _create_encoder(self, base_encoder: nn.Module) -> nn.Module:
        """Create a copy of the base encoder."""
        encoder = type(base_encoder)(**self._get_encoder_args(base_encoder))
        encoder.load_state_dict(base_encoder.state_dict())
        return encoder
    
    def _get_encoder_args(self, encoder: nn.Module) -> Dict[str, Any]:
        """Extract constructor arguments from encoder."""
        args = {}
        if hasattr(encoder, 'input_dim'):
            args['input_dim'] = encoder.input_dim
        if hasattr(encoder, 'hidden_dim'):
            args['hidden_dim'] = encoder.hidden_dim
        if hasattr(encoder, 'output_dim'):
            args['output_dim'] = encoder.output_dim
        if hasattr(encoder, 'input_channels'):
            args['input_channels'] = encoder.input_channels
        return args
    
    def update(self, online_encoder: nn.Module):
        """
        Update momentum encoder with exponential moving average.
        
        Args:
            online_encoder: Online encoder to update from
        """
        for online_param, momentum_param in zip(
            online_encoder.parameters(),
            self.encoder.parameters()
        ):
            momentum_param.data = self.momentum * momentum_param.data + \
                                (1 - self.momentum) * online_param.data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through momentum encoder."""
        return self.encoder(x)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load state dict."""
        self.encoder.load_state_dict(state_dict)
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict."""
        return self.encoder.state_dict()


class FederatedMomentumEncoder(MomentumEncoder):
    """Federated momentum encoder that supports federated updates."""
    
    def __init__(self, base_encoder: nn.Module, momentum: float = 0.999,
                 server_momentum: float = 0.9):
        """
        Initialize federated momentum encoder.
        
        Args:
            base_encoder: Base encoder network
            momentum: Momentum factor for local EMA updates
            server_momentum: Momentum factor for server updates
        """
        super().__init__(base_encoder, momentum)
        
        self.server_momentum = server_momentum
        self.server_encoder = self._create_encoder(base_encoder)
    
    def update_from_server(self, server_state_dict: Dict[str, torch.Tensor]):
        """
        Update from server parameters with momentum.
        
        Args:
            server_state_dict: Server encoder state dict
        """
        for local_param, server_param in zip(
            self.encoder.parameters(),
            server_state_dict.values()
        ):
            local_param.data = self.server_momentum * local_param.data + \
                             (1 - self.server_momentum) * server_param.to(local_param.device)
    
    def sync_with_server(self, server_state_dict: Dict[str, torch.Tensor]):
        """
        Synchronize encoder with server parameters.
        
        Args:
            server_state_dict: Server encoder state dict
        """
        self.encoder.load_state_dict({
            k: v.to(list(self.encoder.parameters())[0].device) 
            for k, v in server_state_dict.items()
        })
    
    def get_server_update(self) -> Dict[str, torch.Tensor]:
        """Get parameters to send to server."""
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}


class MomentumEncoderCoordinator:
    """Coordinator for managing multiple momentum encoders in federation."""
    
    def __init__(self, base_encoder: nn.Module, num_clients: int, 
                 momentum: float = 0.999):
        """
        Initialize momentum encoder coordinator.
        
        Args:
            base_encoder: Base encoder network
            num_clients: Number of clients
            momentum: Momentum factor
        """
        self.base_encoder = base_encoder
        self.num_clients = num_clients
        self.momentum = momentum
        
        self.clients = {}
    
    def create_client_encoder(self, client_id: int) -> FederatedMomentumEncoder:
        """
        Create a momentum encoder for a client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Federated momentum encoder for the client
        """
        encoder = FederatedMomentumEncoder(self.base_encoder, self.momentum)
        self.clients[client_id] = encoder
        return encoder
    
    def aggregate_client_encoders(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate encoder parameters from all clients.
        
        Returns:
            Aggregated encoder state dict
        """
        if not self.clients:
            return self.base_encoder.state_dict()
        
        aggregated = {}
        first_client = list(self.clients.values())[0]
        state_dict = first_client.state_dict()
        
        for key in state_dict:
            params = []
            for client in self.clients.values():
                params.append(client.state_dict()[key])
            
            aggregated[key] = torch.mean(torch.stack(params), dim=0)
        
        return aggregated
    
    def broadcast_to_clients(self, server_state_dict: Dict[str, torch.Tensor]):
        """
        Broadcast server parameters to all clients.
        
        Args:
            server_state_dict: Server encoder state dict
        """
        for client in self.clients.values():
            client.update_from_server(server_state_dict)
    
    def get_client_encoder(self, client_id: int) -> Optional[FederatedMomentumEncoder]:
        """
        Get encoder for a specific client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Client encoder or None
        """
        return self.clients.get(client_id)


class BYOLPredictor(nn.Module):
    """BYOL predictor head for momentum encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = 256):
        """
        Initialize BYOL predictor.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through predictor."""
        return self.predictor(x)


class BYOLMomentumEncoder(FederatedMomentumEncoder):
    """BYOL-specific momentum encoder with predictor."""
    
    def __init__(self, base_encoder: nn.Module, momentum: float = 0.996,
                 predictor_hidden_dim: int = 4096):
        """
        Initialize BYOL momentum encoder.
        
        Args:
            base_encoder: Base encoder network
            momentum: Momentum factor
            predictor_hidden_dim: Hidden dimension for predictor
        """
        super().__init__(base_encoder, momentum)
        
        if hasattr(base_encoder, 'output_dim'):
            output_dim = base_encoder.output_dim
        else:
            output_dim = 256
        
        self.predictor = BYOLPredictor(output_dim, predictor_hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, use_predictor: bool = False) -> torch.Tensor:
        """
        Forward pass with optional predictor.
        
        Args:
            x: Input tensor
            use_predictor: Whether to apply predictor
        
        Returns:
            Encoded features
        """
        features = self.encoder(x)
        
        if use_predictor:
            features = self.predictor(features)
        
        return features
    
    def get_predictor_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get predictor state dict."""
        return {k: v.detach().cpu().clone() for k, v in self.predictor.state_dict().items()}
    
    def load_predictor_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load predictor state dict."""
        self.predictor.load_state_dict({
            k: v.to(list(self.predictor.parameters())[0].device)
            for k, v in state_dict.items()
        })