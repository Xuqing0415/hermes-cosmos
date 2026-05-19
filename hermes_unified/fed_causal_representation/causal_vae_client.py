"""
Causal VAE Client for Federated Causal Representation Learning

Implements variational autoencoder with causal structure for disentangled representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class StructuralEquationModel(nn.Module):
    """Structural Equation Model for causal relationships."""
    
    def __init__(self, num_causal_vars: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_causal_vars = num_causal_vars
        self.hidden_dim = hidden_dim
        
        self.adjacency = nn.Parameter(torch.zeros(num_causal_vars, num_causal_vars))
        
        self.structural_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_causal_vars + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_causal_vars)
        ])
    
    def forward(self, exogenous: torch.Tensor, order: Optional[List[int]] = None) -> torch.Tensor:
        """
        Generate causal variables from exogenous noise.
        
        Args:
            exogenous: Exogenous noise (batch_size x num_causal_vars)
            order: Topological order for causal graph
        
        Returns:
            Causal variables (batch_size x num_causal_vars)
        """
        batch_size = exogenous.size(0)
        causal_vars = torch.zeros(batch_size, self.num_causal_vars, device=exogenous.device)
        
        if order is None:
            order = list(range(self.num_causal_vars))
        
        for i in order:
            parents = (self.adjacency[:, i] > 0.5).float()
            parent_values = causal_vars * parents.unsqueeze(0)
            
            input_vec = torch.cat([parent_values, exogenous[:, i:i+1]], dim=1)
            causal_vars[:, i] = self.structural_functions[i](input_vec).squeeze(-1)
        
        return causal_vars
    
    def get_adjacency_matrix(self, threshold: float = 0.5) -> torch.Tensor:
        """Get binary adjacency matrix."""
        return (torch.sigmoid(self.adjacency) > threshold).float()
    
    def dag_constraint(self) -> torch.Tensor:
        """
        Compute DAG constraint (acyclicity penalty).
        
        Returns:
            DAG constraint loss (should be 0 for DAG)
        """
        adj = torch.sigmoid(self.adjacency)
        d = adj.size(0)
        
        exp_adj = torch.matrix_exp(adj * adj)
        trace = torch.trace(exp_adj)
        
        return trace - d


class CausalEncoder(nn.Module):
    """Encoder for causal representation learning."""
    
    def __init__(self, input_dim: int, num_causal_vars: int = 5, 
                 num_noise_vars: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.num_causal_vars = num_causal_vars
        self.num_noise_vars = num_noise_vars
        self.total_latent = num_causal_vars + num_noise_vars
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.causal_mu = nn.Linear(hidden_dim, num_causal_vars)
        self.causal_logvar = nn.Linear(hidden_dim, num_causal_vars)
        
        self.noise_mu = nn.Linear(hidden_dim, num_noise_vars)
        self.noise_logvar = nn.Linear(hidden_dim, num_noise_vars)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                  torch.Tensor, torch.Tensor]:
        """
        Encode input to causal and noise latent variables.
        
        Args:
            x: Input data (batch_size x input_dim)
        
        Returns:
            causal_z, causal_logvar, noise_z, noise_logvar
        """
        h = self.encoder(x)
        
        causal_mu = self.causal_mu(h)
        causal_logvar = self.causal_logvar(h)
        
        noise_mu = self.noise_mu(h)
        noise_logvar = self.noise_logvar(h)
        
        causal_z = self._reparameterize(causal_mu, causal_logvar)
        noise_z = self._reparameterize(noise_mu, noise_logvar)
        
        return causal_z, causal_logvar, noise_z, noise_logvar
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class CausalDecoder(nn.Module):
    """Decoder for causal representation learning."""
    
    def __init__(self, output_dim: int, num_causal_vars: int = 5,
                 num_noise_vars: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.num_causal_vars = num_causal_vars
        self.num_noise_vars = num_noise_vars
        
        total_latent = num_causal_vars + num_noise_vars
        
        self.decoder = nn.Sequential(
            nn.Linear(total_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, causal_z: torch.Tensor, noise_z: torch.Tensor) -> torch.Tensor:
        """
        Decode from causal and noise latent variables.
        
        Args:
            causal_z: Causal latent variables
            noise_z: Noise latent variables
        
        Returns:
            Reconstructed output
        """
        z = torch.cat([causal_z, noise_z], dim=1)
        return self.decoder(z)


class CausalVAE(nn.Module):
    """Causal Variational Autoencoder."""
    
    def __init__(self, input_dim: int, num_causal_vars: int = 5,
                 num_noise_vars: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = CausalEncoder(input_dim, num_causal_vars, num_noise_vars, hidden_dim)
        self.decoder = CausalDecoder(input_dim, num_causal_vars, num_noise_vars, hidden_dim)
        self.sem = StructuralEquationModel(num_causal_vars, hidden_dim // 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CausalVAE.
        
        Args:
            x: Input data
        
        Returns:
            Dictionary containing reconstruction and latent variables
        """
        causal_z, causal_logvar, noise_z, noise_logvar = self.encoder(x)
        
        exogenous = torch.randn_like(causal_z)
        generated_causal = self.sem(exogenous)
        
        recon = self.decoder(causal_z, noise_z)
        
        return {
            'reconstruction': recon,
            'causal_z': causal_z,
            'causal_logvar': causal_logvar,
            'noise_z': noise_z,
            'noise_logvar': noise_logvar,
            'generated_causal': generated_causal
        }
    
    def compute_loss(self, x: torch.Tensor, recon_weight: float = 1.0,
                     kl_weight: float = 0.1, dag_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute CausalVAE loss.
        
        Args:
            x: Input data
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence
            dag_weight: Weight for DAG constraint
        
        Returns:
            Dictionary of losses
        """
        outputs = self.forward(x)
        
        recon_loss = F.mse_loss(outputs['reconstruction'], x)
        
        causal_kl = -0.5 * torch.mean(1 + outputs['causal_logvar'] - 
                                       outputs['causal_z'].pow(2) - 
                                       outputs['causal_logvar'].exp())
        
        noise_kl = -0.5 * torch.mean(1 + outputs['noise_logvar'] - 
                                      outputs['noise_z'].pow(2) - 
                                      outputs['noise_logvar'].exp())
        
        kl_loss = causal_kl + noise_kl
        
        dag_loss = self.sem.dag_constraint()
        
        total_loss = (recon_weight * recon_loss + 
                      kl_weight * kl_loss + 
                      dag_weight * dag_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'dag_loss': dag_loss
        }


class CausalVAEClient:
    """Federated client for CausalVAE."""
    
    def __init__(self, client_id: int, input_dim: int, num_causal_vars: int = 5,
                 num_noise_vars: int = 3, device: str = 'cpu'):
        self.client_id = client_id
        self.device = device
        
        self.model = CausalVAE(input_dim, num_causal_vars, num_noise_vars).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.environment_id = None
        self.loss_history = []
    
    def set_environment(self, env_id: int):
        """Set environment identifier for this client."""
        self.environment_id = env_id
    
    def local_train(self, dataloader, global_params: Optional[Dict[str, torch.Tensor]] = None,
                    num_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """
        Perform local training.
        
        Args:
            dataloader: Local dataloader
            global_params: Global parameters from server
            num_epochs: Number of local epochs
        
        Returns:
            Parameter updates
        """
        if global_params is not None:
            self._load_shared_params(global_params)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                losses = self.model.compute_loss(x)
                loss = losses['total_loss']
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.loss_history.append(avg_loss)
        
        return self._get_shared_params()
    
    def _get_shared_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters to share with server (SEM and encoder)."""
        params = {
            'sem_adjacency': self.model.sem.adjacency.detach().cpu().clone(),
            'encoder': {k: v.detach().cpu().clone() for k, v in self.model.encoder.state_dict().items()},
            'decoder': {k: v.detach().cpu().clone() for k, v in self.model.decoder.state_dict().items()}
        }
        return params
    
    def _load_shared_params(self, params: Dict[str, torch.Tensor]):
        """Load shared parameters from server."""
        if 'sem_adjacency' in params:
            self.model.sem.adjacency.data = params['sem_adjacency'].to(self.device)
        if 'encoder' in params:
            self.model.encoder.load_state_dict({
                k: v.to(self.device) for k, v in params['encoder'].items()
            })
        if 'decoder' in params:
            self.model.decoder.load_state_dict({
                k: v.to(self.device) for k, v in params['decoder'].items()
            })
    
    def get_causal_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract causal representation from input."""
        self.model.eval()
        with torch.no_grad():
            causal_z, _, _, _ = self.model.encoder(x.to(self.device))
        return causal_z
    
    def get_causal_graph(self, threshold: float = 0.5) -> np.ndarray:
        """Get learned causal graph as numpy array."""
        return self.model.sem.get_adjacency_matrix(threshold).cpu().numpy()
    
    def counterfactual(self, x: torch.Tensor, intervention: Dict[int, float]) -> torch.Tensor:
        """
        Perform counterfactual reasoning.
        
        Args:
            x: Observed data
            intervention: Dictionary of {variable_index: value} for intervention
        
        Returns:
            Counterfactual outcome
        """
        self.model.eval()
        
        with torch.no_grad():
            causal_z, causal_logvar, noise_z, noise_logvar = self.model.encoder(x.to(self.device))
            
            for var_idx, value in intervention.items():
                causal_z[:, var_idx] = value
            
            counterfactual = self.model.decoder(causal_z, noise_z)
        
        return counterfactual


def generate_causal_data(num_samples: int = 1000, num_causal_vars: int = 5,
                         num_noise_vars: int = 3, env_id: int = 0) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate synthetic causal data.
    
    Args:
        num_samples: Number of samples
        num_causal_vars: Number of causal variables
        num_noise_vars: Number of noise variables
        env_id: Environment identifier (affects distribution)
    
    Returns:
        Data tensor and true adjacency matrix
    """
    true_adj = np.zeros((num_causal_vars, num_causal_vars))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1
    true_adj[0, 3] = 1
    true_adj[3, 4] = 1
    
    exogenous = np.random.randn(num_samples, num_causal_vars)
    
    env_shift = env_id * 0.5
    exogenous[:, 0] += env_shift
    
    causal_vars = np.zeros((num_samples, num_causal_vars))
    causal_vars[:, 0] = exogenous[:, 0]
    causal_vars[:, 1] = 0.5 * causal_vars[:, 0] + exogenous[:, 1]
    causal_vars[:, 2] = 0.3 * causal_vars[:, 1] + exogenous[:, 2]
    causal_vars[:, 3] = 0.4 * causal_vars[:, 0] + exogenous[:, 3]
    causal_vars[:, 4] = 0.6 * causal_vars[:, 3] + exogenous[:, 4]
    
    noise_vars = np.random.randn(num_samples, num_noise_vars)
    
    data = np.concatenate([causal_vars, noise_vars], axis=1)
    
    return torch.tensor(data, dtype=torch.float32), true_adj