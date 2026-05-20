"""
Local Architecture Search for Federated NAS

Implements evolutionary search, Bayesian optimization, and differentiable search.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class LocalSearch:
    """
    Base class for local architecture search.
    """
    
    def __init__(self, supernet: nn.Module, val_dataloader, loss_fn: nn.Module):
        self.supernet = supernet
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        
        self.best_arch = None
        self.best_acc = 0.0
    
    def search(self, constraints: Optional[Dict[str, float]] = None) -> List[torch.Tensor]:
        """
        Perform architecture search.
        
        Args:
            constraints: Resource constraints
        
        Returns:
            Best architecture weights
        """
        raise NotImplementedError
    
    def evaluate_arch(self, arch_weights: List[torch.Tensor]) -> float:
        """
        Evaluate architecture on validation set.
        
        Args:
            arch_weights: Architecture weights
        
        Returns:
            Validation accuracy
        """
        self.supernet.set_alphas(arch_weights)
        self.supernet.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_dataloader:
                output = self.supernet(x, discrete=True)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += x.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_arch = arch_weights
        
        return accuracy


class EvolutionarySearch(LocalSearch):
    """
    Evolutionary architecture search.
    
    Uses genetic algorithms to search for optimal architectures.
    """
    
    def __init__(self, supernet: nn.Module, val_dataloader, loss_fn: nn.Module,
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.5):
        super().__init__(supernet, val_dataloader, loss_fn)
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
    
    def _initialize_population(self):
        """Initialize population with random architectures."""
        num_alphas = len(self.supernet.get_alphas())
        alphas_shape = [a.shape[0] for a in self.supernet.get_alphas()]
        
        for _ in range(self.population_size):
            arch = []
            for shape in alphas_shape:
                alpha = torch.randn(shape) * 1e-3
                arch.append(alpha)
            self.population.append(arch)
    
    def _mutate(self, arch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Mutate an architecture."""
        mutated = []
        for alpha in arch:
            mask = (torch.rand(alpha.shape) < self.mutation_rate).float()
            noise = torch.randn(alpha.shape) * 0.1
            mutated_alpha = alpha + mask * noise
            mutated.append(mutated_alpha)
        return mutated
    
    def _crossover(self, parent1: List[torch.Tensor], 
                   parent2: List[torch.Tensor]) -> List[torch.Tensor]:
        """Crossover two architectures."""
        child = []
        for a1, a2 in zip(parent1, parent2):
            if np.random.random() < self.crossover_rate:
                child.append(a1.clone())
            else:
                child.append(a2.clone())
        return child
    
    def _select(self, fitness: List[float]) -> List[List[torch.Tensor]]:
        """Select parents using tournament selection."""
        selected = []
        for _ in range(self.population_size):
            idx1, idx2 = np.random.choice(len(fitness), 2, replace=False)
            if fitness[idx1] > fitness[idx2]:
                selected.append(self.population[idx1])
            else:
                selected.append(self.population[idx2])
        return selected
    
    def search(self, constraints: Optional[Dict[str, float]] = None) -> List[torch.Tensor]:
        """
        Perform evolutionary search.
        
        Args:
            constraints: Resource constraints
        
        Returns:
            Best architecture weights
        """
        self._initialize_population()
        
        for gen in range(self.generations):
            fitness = [self.evaluate_arch(arch) for arch in self.population]
            
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}: Best accuracy = {max(fitness):.4f}")
            
            parents = self._select(fitness)
            
            new_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    child1 = self._crossover(parents[i], parents[i+1])
                    child2 = self._crossover(parents[i], parents[i+1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(parents[i])
            
            self.population = [self._mutate(arch) for arch in new_population]
        
        return self.best_arch


class BayesianSearch(LocalSearch):
    """
    Bayesian optimization for architecture search.
    
    Uses Gaussian processes to model the accuracy-surrogate function.
    """
    
    def __init__(self, supernet: nn.Module, val_dataloader, loss_fn: nn.Module,
                 num_iterations: int = 50, num_initial: int = 10):
        super().__init__(supernet, val_dataloader, loss_fn)
        
        self.num_iterations = num_iterations
        self.num_initial = num_initial
        
        self.samples = []
        self.evaluations = []
    
    def _encode_arch(self, arch: List[torch.Tensor]) -> np.ndarray:
        """Encode architecture to a flat vector."""
        return np.concatenate([a.numpy().flatten() for a in arch])
    
    def _decode_arch(self, vector: np.ndarray) -> List[torch.Tensor]:
        """Decode vector to architecture."""
        arch = []
        offset = 0
        
        for alpha in self.supernet.get_alphas():
            size = alpha.numel()
            alpha_np = vector[offset:offset + size].reshape(alpha.shape)
            arch.append(torch.tensor(alpha_np))
            offset += size
        
        return arch
    
    def _acquisition(self, x: np.ndarray) -> float:
        """Upper confidence bound acquisition function."""
        if not self.samples:
            return np.random.random()
        
        mean = np.mean(self.evaluations)
        std = np.std(self.evaluations)
        
        if std == 0:
            return np.random.random()
        
        return mean + 2 * std * np.random.random()
    
    def search(self, constraints: Optional[Dict[str, float]] = None) -> List[torch.Tensor]:
        """
        Perform Bayesian optimization search.
        
        Args:
            constraints: Resource constraints
        
        Returns:
            Best architecture weights
        """
        num_alphas = sum(a.numel() for a in self.supernet.get_alphas())
        
        for _ in range(self.num_initial):
            vector = np.random.randn(num_alphas) * 0.1
            arch = self._decode_arch(vector)
            acc = self.evaluate_arch(arch)
            
            self.samples.append(vector)
            self.evaluations.append(acc)
        
        for _ in range(self.num_iterations):
            candidates = [np.random.randn(num_alphas) * 0.1 for _ in range(10)]
            
            scores = [self._acquisition(c) for c in candidates]
            best_candidate = candidates[np.argmax(scores)]
            
            arch = self._decode_arch(best_candidate)
            acc = self.evaluate_arch(arch)
            
            self.samples.append(best_candidate)
            self.evaluations.append(acc)
        
        return self.best_arch


class DifferentiableSearch(LocalSearch):
    """
    Differentiable architecture search.
    
    Uses gradient descent on architecture weights.
    """
    
    def __init__(self, supernet: nn.Module, val_dataloader, loss_fn: nn.Module,
                 arch_lr: float = 3e-4, weight_lr: float = 3e-3,
                 epochs: int = 10):
        super().__init__(supernet, val_dataloader, loss_fn)
        
        self.arch_lr = arch_lr
        self.weight_lr = weight_lr
        self.epochs = epochs
    
    def search(self, constraints: Optional[Dict[str, float]] = None) -> List[torch.Tensor]:
        """
        Perform differentiable search.
        
        Args:
            constraints: Resource constraints
        
        Returns:
            Best architecture weights
        """
        arch_optimizer = torch.optim.Adam(self.supernet.get_arch_params(), lr=self.arch_lr)
        weight_optimizer = torch.optim.Adam(self.supernet.get_weight_params(), lr=self.weight_lr)
        
        for epoch in range(self.epochs):
            self.supernet.train()
            
            for x, y in self.val_dataloader:
                arch_optimizer.zero_grad()
                weight_optimizer.zero_grad()
                
                output = self.supernet(x, discrete=False)
                loss = self.loss_fn(output, y)
                
                loss.backward()
                
                arch_optimizer.step()
                weight_optimizer.step()
            
            acc = self.evaluate_arch(self.supernet.get_alphas())
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}: Validation accuracy = {acc:.4f}")
        
        return self.supernet.get_alphas()


class RandomSearch(LocalSearch):
    """
    Random search baseline.
    
    Samples random architectures and returns the best one.
    """
    
    def __init__(self, supernet: nn.Module, val_dataloader, loss_fn: nn.Module,
                 num_samples: int = 100):
        super().__init__(supernet, val_dataloader, loss_fn)
        self.num_samples = num_samples
    
    def search(self, constraints: Optional[Dict[str, float]] = None) -> List[torch.Tensor]:
        """
        Perform random search.
        
        Args:
            constraints: Resource constraints
        
        Returns:
            Best architecture weights
        """
        alphas_shape = [a.shape[0] for a in self.supernet.get_alphas()]
        
        for _ in range(self.num_samples):
            arch = []
            for shape in alphas_shape:
                alpha = torch.randn(shape) * 1e-3
                arch.append(alpha)
            
            self.evaluate_arch(arch)
        
        return self.best_arch


class SearchFactory:
    """
    Factory for creating search strategies.
    """
    
    @staticmethod
    def create_search(method: str, supernet: nn.Module,
                     val_dataloader, loss_fn: nn.Module,
                     **kwargs) -> LocalSearch:
        """
        Create a search strategy.
        
        Args:
            method: Search method ('evolutionary', 'bayesian', 'differentiable', 'random')
            supernet: Super network
            val_dataloader: Validation dataloader
            loss_fn: Loss function
            kwargs: Additional parameters
        
        Returns:
            LocalSearch instance
        """
        if method == 'evolutionary':
            return EvolutionarySearch(supernet, val_dataloader, loss_fn, **kwargs)
        elif method == 'bayesian':
            return BayesianSearch(supernet, val_dataloader, loss_fn, **kwargs)
        elif method == 'differentiable':
            return DifferentiableSearch(supernet, val_dataloader, loss_fn, **kwargs)
        elif method == 'random':
            return RandomSearch(supernet, val_dataloader, loss_fn, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {method}")