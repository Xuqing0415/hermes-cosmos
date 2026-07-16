"""
Federated Online Learning Main Module

Implements the complete federated online learning framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional

from .online_client import OnlineClient
from .online_server import OnlineServer, FederatedOnlineEvaluator
from .drift_detector import CUSUMDetector, ADWINDetector, DriftDetectorEnsemble
from .fomaml import FederatedOnlineMetaLearner


class FederatedOnlineLearning:
    """Federated online learning framework."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 num_clients: int = 10, lr: float = 0.01,
                 batch_size: int = 10, use_fomaml: bool = False):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.lr = lr
        self.batch_size = batch_size
        self.use_fomaml = use_fomaml
        
        if use_fomaml:
            self.meta_learner = FederatedOnlineMetaLearner(
                model_class, loss_fn, num_clients,
                inner_lr=lr, meta_lr=lr * 0.1
            )
        else:
            self.server = OnlineServer(model_class())
            self.clients: Dict[int, OnlineClient] = {}
            self._initialize_clients()
        
        self.results = {
            'regret': [],
            'losses': [],
            'drift_detections': [],
            'global_updates': [],
            'communication_cost': []
        }
    
    def _initialize_clients(self):
        """Initialize online clients."""
        for client_id in range(self.num_clients):
            model = self.model_class()
            model.load_state_dict(self.server.get_global_model())
            
            detector = DriftDetectorEnsemble([
                CUSUMDetector(delta=0.01, threshold=10.0),
                ADWINDetector(delta=0.001)
            ])
            
            client = OnlineClient(
                client_id=client_id,
                model=model,
                loss_fn=self.loss_fn,
                lr=self.lr,
                batch_size=self.batch_size,
                drift_detector=detector
            )
            
            self.clients[client_id] = client
    
    def process_stream(self, stream_data: List[Dict[int, Any]],
                      meta_iterations: int = 1):
        """
        Process streaming data.
        
        Args:
            stream_data: List of client data for each time step
            meta_iterations: Number of meta-iterations per batch
        """
        if self.use_fomaml:
            self._process_fomaml_stream(stream_data)
        else:
            self._process_standard_stream(stream_data)
    
    def _process_standard_stream(self, stream_data: List[Dict[int, Any]]):
        """Process stream using standard online federated learning."""
        regret = 0.0
        
        for step, client_data in enumerate(stream_data):
            for client_id, (x, y) in client_data.items():
                if client_id in self.clients:
                    result = self.clients[client_id].process_sample(x, y)
                    
                    if 'gradients' in result:
                        self.server.receive_gradient(client_id, result['gradients'])
                        self.results['communication_cost'].append(1)
                    
                    if 'drift_detected' in result:
                        self.server.receive_drift_alert(result['alert'].to_dict())
                        self.results['drift_detections'].append(step)
            
            updated = self.server.update_global_model(min_clients=1)
            
            if updated:
                global_model = self.server.get_global_model()
                for client in self.clients.values():
                    client.update_global_model(global_model)
                
                self.results['global_updates'].append(self.server.global_updates)
    
    def _process_fomaml_stream(self, stream_data: List[Dict[int, Any]]):
        """Process stream using FOMAML."""
        for step, client_data in enumerate(stream_data):
            if step % 10 == 0:
                meta_data = {}
                for client_id, (x, y) in client_data.items():
                    if client_id < self.num_clients:
                        meta_data[client_id] = (x, y)
                
                if meta_data:
                    self.meta_learner.run_meta_iteration(meta_data)
                    self.results['global_updates'].append(self.meta_learner.get_summary()['meta_updates'])
    
    def evaluate(self, dataloader) -> float:
        """Evaluate global model."""
        if self.use_fomaml:
            return self.meta_learner.evaluate(dataloader)
        else:
            evaluator = FederatedOnlineEvaluator(self.server)
            return evaluator.evaluate(dataloader, self.loss_fn)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.use_fomaml:
            return self.meta_learner.get_summary()
        else:
            return {
                'global_updates': self.server.get_server_stats()['global_updates'],
                'drift_detections': len(self.results['drift_detections']),
                'communication_cost': sum(self.results['communication_cost'])
            }


def generate_streaming_data(num_samples: int = 1000, 
                           num_clients: int = 10,
                           drift_interval: int = 500) -> List[Dict[int, Any]]:
    """
    Generate synthetic streaming data with concept drift.
    
    Args:
        num_samples: Total number of samples
        num_clients: Number of clients
        drift_interval: Interval between drifts
    
    Returns:
        List of client data dictionaries
    """
    stream = []
    
    for i in range(num_samples):
        client_data = {}
        
        for client_id in range(num_clients):
            x = torch.randn(1, 10)
            
            if i < drift_interval:
                y = torch.tensor([1 if x[0, 0] > 0 else 0], dtype=torch.float32)
            else:
                y = torch.tensor([1 if x[0, 1] > 0 else 0], dtype=torch.float32)
            
            client_data[client_id] = (x, y)
        
        stream.append(client_data)
    
    return stream


def run_online_demo():
    """Run federated online learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED ONLINE LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"    torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.federated_online.drift_detector import (
                CUSUMDetector, ADWINDetector, DriftDetectorEnsemble
            )
            print("    Drift detectors imported")
        except Exception as e:
            print(f"    Drift detectors import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.online_client import OnlineClient
            print("    OnlineClient imported")
        except Exception as e:
            print(f"    OnlineClient import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.online_server import OnlineServer
            print("    OnlineServer imported")
        except Exception as e:
            print(f"    OnlineServer import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.fomaml import FOMAMLClient, FOMAMLServer
            print("    FOMAML components imported")
        except Exception as e:
            print(f"    FOMAML import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing DriftDetector...")
            detector = CUSUMDetector(threshold=5.0)
            for i in range(100):
                detector.update(0.1)
            print("    CUSUM detector initialized")
        except Exception as e:
            print(f"    DriftDetector failed: {e}")
        
        try:
            print("   Testing OnlineClient...")
            model = torch.nn.Linear(10, 1)
            client = OnlineClient(0, model, torch.nn.MSELoss())
            print(f"    OnlineClient created")
        except Exception as e:
            print(f"    OnlineClient failed: {e}")
        
        try:
            print("   Testing OnlineServer...")
            server = OnlineServer(torch.nn.Linear(10, 1))
            print(f"    OnlineServer created")
        except Exception as e:
            print(f"    OnlineServer failed: {e}")
        
        try:
            print("   Testing FederatedOnlineLearning...")
            fol = FederatedOnlineLearning(
                model_class=torch.nn.Linear,
                loss_fn=torch.nn.MSELoss(),
                num_clients=3,
                batch_size=5
            )
            print(f"    FederatedOnlineLearning created with {len(fol.clients)} clients")
        except Exception as e:
            print(f"    FederatedOnlineLearning failed: {e}")
        
        print("\n Federated Online Learning demo completed!")
        
    except ImportError as e:
        print(f" Import error: {e}")
    except Exception as e:
        print(f" Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_online_demo()