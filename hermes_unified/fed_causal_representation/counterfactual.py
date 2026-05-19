"""
Counterfactual Reasoning for Federated Causal Representation Learning

Implements counterfactual inference and intervention models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class InterventionModel(nn.Module):
    """Model for performing interventions on causal variables."""
    
    def __init__(self, num_causal_vars: int = 5, num_actions: int = 3):
        super().__init__()
        self.num_causal_vars = num_causal_vars
        self.num_actions = num_actions
        
        self.intervention_net = nn.Sequential(
            nn.Linear(num_causal_vars + num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, num_causal_vars)
        )
    
    def forward(self, causal_vars: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Apply intervention to causal variables.
        
        Args:
            causal_vars: Original causal variables (batch_size x num_causal_vars)
            action: Intervention action (batch_size x num_actions)
        
        Returns:
            Intervened causal variables
        """
        input_vec = torch.cat([causal_vars, action], dim=1)
        intervention_effect = self.intervention_net(input_vec)
        
        intervened_vars = causal_vars + intervention_effect
        return intervened_vars
    
    def do_intervention(self, causal_vars: torch.Tensor, var_idx: int, 
                        value: float) -> torch.Tensor:
        """
        Perform do-intervention on a specific variable.
        
        Args:
            causal_vars: Original causal variables
            var_idx: Index of variable to intervene on
            value: Value to set
        
        Returns:
            Intervened causal variables
        """
        intervened = causal_vars.clone()
        intervened[:, var_idx] = value
        return intervened


class CounterfactualReasoner(nn.Module):
    """Module for counterfactual reasoning."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, 
                 sem: nn.Module, num_causal_vars: int = 5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sem = sem
        self.num_causal_vars = num_causal_vars
    
    def abduction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abduction step: infer exogenous noise from observation.
        
        Args:
            x: Observed data
        
        Returns:
            Causal variables and exogenous noise
        """
        with torch.no_grad():
            causal_z, causal_logvar, noise_z, noise_logvar = self.encoder(x)
        
        exogenous = torch.randn_like(causal_z)
        
        return causal_z, exogenous
    
    def action(self, causal_z: torch.Tensor, intervention: Dict[int, float]) -> torch.Tensor:
        """
        Action step: apply intervention to causal variables.
        
        Args:
            causal_z: Original causal variables
            intervention: Dictionary of {var_idx: value}
        
        Returns:
            Intervened causal variables
        """
        intervened_z = causal_z.clone()
        for var_idx, value in intervention.items():
            if 0 <= var_idx < self.num_causal_vars:
                intervened_z[:, var_idx] = value
        return intervened_z
    
    def prediction(self, intervened_z: torch.Tensor, noise_z: torch.Tensor) -> torch.Tensor:
        """
        Prediction step: generate counterfactual outcome.
        
        Args:
            intervened_z: Intervened causal variables
            noise_z: Noise variables
        
        Returns:
            Counterfactual reconstruction
        """
        return self.decoder(intervened_z, noise_z)
    
    def counterfactual(self, x: torch.Tensor, intervention: Dict[int, float]) -> torch.Tensor:
        """
        Full counterfactual inference pipeline.
        
        Args:
            x: Observed data
            intervention: Dictionary of {var_idx: value}
        
        Returns:
            Counterfactual outcome
        """
        causal_z, exogenous = self.abduction(x)
        intervened_z = self.action(causal_z, intervention)
        
        with torch.no_grad():
            noise_z = torch.zeros(x.size(0), 3, device=x.device)
        
        counterfactual_x = self.prediction(intervened_z, noise_z)
        
        return counterfactual_x
    
    def average_treatment_effect(self, x: torch.Tensor, var_idx: int,
                                  treatment_value: float, 
                                  control_value: float) -> torch.Tensor:
        """
        Compute Average Treatment Effect (ATE).
        
        Args:
            x: Observed data
            var_idx: Variable index to intervene on
            treatment_value: Treatment value
            control_value: Control value
        
        Returns:
            ATE estimate
        """
        treatment_outcome = self.counterfactual(x, {var_idx: treatment_value})
        control_outcome = self.counterfactual(x, {var_idx: control_value})
        
        ate = torch.mean(treatment_outcome - control_outcome, dim=0)
        return ate
    
    def individual_treatment_effect(self, x: torch.Tensor, var_idx: int,
                                     treatment_value: float,
                                     control_value: float) -> torch.Tensor:
        """
        Compute Individual Treatment Effect (ITE).
        
        Args:
            x: Observed data
            var_idx: Variable index to intervene on
            treatment_value: Treatment value
            control_value: Control value
        
        Returns:
            ITE estimates for each sample
        """
        treatment_outcome = self.counterfactual(x, {var_idx: treatment_value})
        control_outcome = self.counterfactual(x, {var_idx: control_value})
        
        ite = treatment_outcome - control_outcome
        return ite


class CausalEffectEstimator:
    """Estimator for various causal effects."""
    
    def __init__(self, reasoner: CounterfactualReasoner):
        self.reasoner = reasoner
    
    def estimate_ate(self, data: torch.Tensor, treatment_var: int,
                     outcome_var: int, treatment_value: float = 1.0,
                     control_value: float = 0.0) -> float:
        """
        Estimate Average Treatment Effect.
        
        Args:
            data: Observed data
            treatment_var: Treatment variable index
            outcome_var: Outcome variable index
            treatment_value: Value for treatment group
            control_value: Value for control group
        
        Returns:
            ATE estimate
        """
        ate = self.reasoner.average_treatment_effect(
            data, treatment_var, treatment_value, control_value
        )
        return float(ate[:, outcome_var].mean())
    
    def estimate_ite(self, data: torch.Tensor, treatment_var: int,
                     outcome_var: int, treatment_value: float = 1.0,
                     control_value: float = 0.0) -> np.ndarray:
        """
        Estimate Individual Treatment Effects.
        
        Args:
            data: Observed data
            treatment_var: Treatment variable index
            outcome_var: Outcome variable index
            treatment_value: Value for treatment group
            control_value: Value for control group
        
        Returns:
            ITE estimates
        """
        ite = self.reasoner.individual_treatment_effect(
            data, treatment_var, treatment_value, control_value
        )
        return ite[:, outcome_var].cpu().numpy()
    
    def estimate_cate(self, data: torch.Tensor, treatment_var: int,
                      outcome_var: int, subgroup_mask: torch.Tensor,
                      treatment_value: float = 1.0,
                      control_value: float = 0.0) -> float:
        """
        Estimate Conditional Average Treatment Effect.
        
        Args:
            data: Observed data
            treatment_var: Treatment variable index
            outcome_var: Outcome variable index
            subgroup_mask: Boolean mask for subgroup
            treatment_value: Value for treatment group
            control_value: Value for control group
        
        Returns:
            CATE estimate
        """
        subgroup_data = data[subgroup_mask]
        
        if subgroup_data.size(0) == 0:
            return 0.0
        
        ate = self.reasoner.average_treatment_effect(
            subgroup_data, treatment_var, treatment_value, control_value
        )
        return float(ate[:, outcome_var].mean())


class PersonalizedInterventionRecommender:
    """Recommender for personalized interventions."""
    
    def __init__(self, reasoner: CounterfactualReasoner, num_vars: int = 5):
        self.reasoner = reasoner
        self.num_vars = num_vars
    
    def recommend_intervention(self, x: torch.Tensor, target_var: int,
                               target_value: float, 
                               actionable_vars: List[int]) -> Dict[int, float]:
        """
        Recommend intervention to achieve target outcome.
        
        Args:
            x: Observed data
            target_var: Target variable index
            target_value: Desired target value
            actionable_vars: Variables that can be intervened on
        
        Returns:
            Recommended intervention {var_idx: value}
        """
        best_intervention = {}
        best_distance = float('inf')
        
        for var_idx in actionable_vars:
            for test_value in np.linspace(-2, 2, 20):
                intervention = {var_idx: test_value}
                counterfactual = self.reasoner.counterfactual(x, intervention)
                
                distance = torch.abs(counterfactual[:, target_var] - target_value).mean().item()
                
                if distance < best_distance:
                    best_distance = distance
                    best_intervention = {var_idx: test_value}
        
        return best_intervention
    
    def rank_interventions(self, x: torch.Tensor, target_var: int,
                           target_value: float, 
                           actionable_vars: List[int]) -> List[Tuple[int, float, float]]:
        """
        Rank possible interventions by effectiveness.
        
        Args:
            x: Observed data
            target_var: Target variable index
            target_value: Desired target value
            actionable_vars: Variables that can be intervened on
        
        Returns:
            List of (var_idx, value, effectiveness) sorted by effectiveness
        """
        results = []
        
        for var_idx in actionable_vars:
            for test_value in np.linspace(-2, 2, 10):
                intervention = {var_idx: test_value}
                counterfactual = self.reasoner.counterfactual(x, intervention)
                
                effectiveness = -torch.abs(counterfactual[:, target_var] - target_value).mean().item()
                
                results.append((var_idx, test_value, effectiveness))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results


def evaluate_counterfactual_accuracy(reasoner: CounterfactualReasoner,
                                     test_data: torch.Tensor,
                                     true_interventions: List[Dict[int, float]],
                                     true_outcomes: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate counterfactual prediction accuracy.
    
    Args:
        reasoner: Counterfactual reasoner
        test_data: Test data
        true_interventions: True interventions applied
        true_outcomes: True counterfactual outcomes
    
    Returns:
        Evaluation metrics
    """
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    
    for i, (x, intervention, true_outcome) in enumerate(zip(test_data, true_interventions, true_outcomes)):
        x_batch = x.unsqueeze(0)
        predicted = reasoner.counterfactual(x_batch, intervention)
        
        mse = F.mse_loss(predicted.squeeze(0), true_outcome).item()
        mae = F.l1_loss(predicted.squeeze(0), true_outcome).item()
        
        total_mse += mse
        total_mae += mae
        num_samples += 1
    
    return {
        'mse': total_mse / num_samples if num_samples > 0 else 0,
        'mae': total_mae / num_samples if num_samples > 0 else 0
    }