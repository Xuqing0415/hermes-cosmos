"""
Unified Entry Point for Federated Learning Experiments

Supports running experiments in eleven directions:
1. FedPEFT - Federated Parameter-Efficient Fine-Tuning for LLMs
2. Federated Transfer Learning
3. Federated Quantum Learning
4. Federated Multimodal Learning
5. Federated Self-Supervised Learning (FedSimCLR)
6. Federated Causal Representation Learning
7. Federated Linear Mode Connectivity
8. Topology-Aware Federated Learning
9. Federated Online Learning
10. Federated Shapley Value Learning
11. Federated Neural Architecture Search (FedNAS)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_fed_peft_demo():
    """Run FedPEFT demonstration."""
    print("\n" + "=" * 70)
    print("  FEDPEFT - FEDERATED PARAMETER-EFFICIENT FINE-TUNING")
    print("=" * 70)
    
    try:
        from hermes_unified.llm_finetune.fed_peft import (
            FedBitFitModule,
            FedPrefixModule,
            AdaptiveLoRAModule,
            compare_peft_methods
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading model: {model_name}")
        
        model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
        
        print("\n1. FedBitFit (Bias-only tuning):")
        bitfit = FedBitFitModule(model)
        print(f"   Trainable params: {bitfit.count_trainable_parameters():,}")
        
        print("\n2. FedPrefix (Prefix tuning):")
        prefix = FedPrefixModule(model, tokenizer, prefix_len=10)
        print(f"   Trainable params: {prefix.count_trainable_parameters():,}")
        
        print("\n3. Adaptive LoRA:")
        lora = AdaptiveLoRAModule(model, tokenizer)
        print(f"   Initial rank: {lora.current_rank}")
        lora.set_rank(16)
        print(f"   After rank adjustment: {lora.current_rank}")
        
        print("\n" + "-" * 70)
        print("  PEFT Method Comparison")
        print("-" * 70)
        sample_input = tokenizer("Hello, world!", return_tensors="pt")
        results = compare_peft_methods(model, tokenizer, None, sample_input)
        
        print("\n✓ FedPEFT demo completed!")
        
    except Exception as e:
        print(f"⚠️ Error in FedPEFT demo: {e}")
        print("Make sure transformers and peft are installed.")


def run_transfer_learning_demo():
    """Run federated transfer learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED TRANSFER LEARNING")
    print("=" * 70)
    
    try:
        from hermes_unified.transfer_learning.domain_adaptation import (
            MMDLoss,
            DomainAdversarialAdaptor,
            DomainAdaptationServer,
            DomainAdaptationClient
        )
        from hermes_unified.transfer_learning.knowledge_distillation import (
            FedDistillationServer,
            FedDistillationClient,
            DistillationLoss
        )
        import torch
        import torch.nn as nn
        
        print("\n1. Testing MMD Loss:")
        mmd = MMDLoss(kernel_type='rbf')
        source = torch.randn(32, 20)
        target = torch.randn(32, 20)
        loss = mmd(source, target)
        print(f"   MMD Loss: {loss.item():.4f}")
        
        print("\n2. Testing Domain Adversarial Adaptor:")
        feature_extractor = nn.Linear(20, 10)
        classifier = nn.Linear(10, 5)
        adaptor = DomainAdversarialAdaptor(feature_extractor, classifier, 10)
        
        source_data = (torch.randn(16, 20), torch.randint(0, 5, (16,)))
        target_data = (torch.randn(16, 20),)
        losses = adaptor.train_step(source_data, target_data)
        print(f"   Training losses computed successfully")
        
        print("\n3. Testing Federated Distillation:")
        teacher = nn.Linear(20, 5)
        student = nn.Linear(20, 5)
        
        dataset = [(torch.randn(20), torch.randint(0, 5, ())) for _ in range(100)]
        client = FedDistillationClient(student, teacher, dataset, device='cpu')
        client.local_train(teacher.state_dict(), num_epochs=1)
        print(f"   Distillation training completed")
        
        print("\n✓ Transfer Learning demo completed!")
        
    except Exception as e:
        print(f"⚠️ Error in Transfer Learning demo: {e}")


def run_quantum_learning_demo():
    """Run federated quantum learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED QUANTUM LEARNING")
    print("=" * 70)
    
    try:
        from hermes_unified.q_fl.fed_qnn import (
            QuantumCircuit,
            QuantumClassifier,
            FedQNNClient,
            FedQNNServer,
            generate_quantum_dataset
        )
        
        print("\n1. Testing Quantum Circuit:")
        circuit = QuantumCircuit(num_qubits=4, num_layers=2)
        params = circuit.circuit.params if hasattr(circuit.circuit, 'params') else None
        print(f"   Qubits: {circuit.num_qubits}, Layers: {circuit.num_layers}")
        
        print("\n2. Testing Quantum Classifier:")
        classifier = QuantumClassifier(num_qubits=4, num_classes=2)
        print(f"   Parameters: {classifier.count_parameters():,}")
        
        print("\n3. Testing FedQNN Client:")
        client = FedQNNClient(client_id=0, num_qubits=4, num_classes=2)
        dataset = generate_quantum_dataset(num_samples=20)
        client.set_data(dataset)
        print(f"   Client created with {len(dataset)} samples")
        
        print("\n4. Testing FedQNN Server:")
        server = FedQNNServer(num_qubits=4, num_classes=2)
        print(f"   Server initialized")
        
        print("\n✓ Quantum Learning demo completed!")
        
    except ImportError:
        print("⚠️ PennyLane not installed. Skipping quantum demo.")
        print("Install with: pip install pennylane")
    except Exception as e:
        print(f"⚠️ Error in Quantum Learning demo: {e}")


def run_multimodal_demo():
    """Run federated multimodal learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED MULTIMODAL LEARNING")
    print("=" * 70)
    
    print("\n1. Checking environment...")
    import sys
    print(f"   Python version: {sys.version}")
    
    print("\n2. Checking torch installation...")
    try:
        import torch
        print(f"   ✓ torch version: {torch.__version__}")
    except ImportError as e:
        print(f"   ✗ torch not installed: {e}")
        print("   Please install torch first: pip install torch")
        return
    
    print("\n3. Testing Core Components Import:")
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import ImageEncoder
        print("   ✓ ImageEncoder imported")
    except Exception as e:
        print(f"   ✗ ImageEncoder import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import TextEncoder
        print("   ✓ TextEncoder imported")
    except Exception as e:
        print(f"   ✗ TextEncoder import failed: {e}")
        return
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import TabularEncoder
        print("   ✓ TabularEncoder imported")
    except Exception as e:
        print(f"   ✗ TabularEncoder import failed: {e}")
        return
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import SharedPrivateEncoder
        print("   ✓ SharedPrivateEncoder imported")
    except Exception as e:
        print(f"   ✗ SharedPrivateEncoder import failed: {e}")
        return
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import MultimodalFusion
        print("   ✓ MultimodalFusion imported")
    except Exception as e:
        print(f"   ✗ MultimodalFusion import failed: {e}")
        return
    
    try:
        from hermes_unified.fed_multimodal.multimodal_client import MultimodalFedClient
        print("   ✓ MultimodalFedClient imported")
    except Exception as e:
        print(f"   ✗ MultimodalFedClient import failed: {e}")
        return
    
    print("\n4. Testing Component Functionality:")
    
    try:
        print("   Testing ImageEncoder...")
        encoder = ImageEncoder()
        dummy = torch.randn(2, 1, 28, 28)
        output = encoder(dummy)
        print(f"   ✓ ImageEncoder: {output.shape}")
    except Exception as e:
        print(f"   ✗ ImageEncoder failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("   Testing TextEncoder...")
        encoder = TextEncoder()
        dummy = torch.randint(0, 1000, (2, 10))
        output = encoder(dummy)
        print(f"   ✓ TextEncoder: {output.shape}")
    except Exception as e:
        print(f"   ✗ TextEncoder failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("   Testing TabularEncoder...")
        encoder = TabularEncoder()
        dummy = torch.randn(2, 10)
        output = encoder(dummy)
        print(f"   ✓ TabularEncoder: {output.shape}")
    except Exception as e:
        print(f"   ✗ TabularEncoder failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("   Testing MultimodalFusion...")
        fusion = MultimodalFusion(num_modalities=3)
        f1 = torch.randn(2, 64)
        f2 = torch.randn(2, 64)
        f3 = torch.randn(2, 64)
        output = fusion([f1, f2, f3])
        print(f"   ✓ MultimodalFusion: {output.shape}")
    except Exception as e:
        print(f"   ✗ MultimodalFusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("   Testing MultimodalFedClient...")
        client = MultimodalFedClient(client_id=0, available_modalities=['image'], num_classes=10)
        print(f"   ✓ MultimodalFedClient created")
    except Exception as e:
        print(f"   ✗ MultimodalFedClient failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Multimodal Learning demo completed!")


def run_self_supervised_demo():
    """Run federated self-supervised learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED SELF-SUPERVISED LEARNING (FedSimCLR)")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_self_supervised.contrastive_client import SimCLRClient
            print("   ✓ SimCLRClient imported")
        except Exception as e:
            print(f"   ✗ SimCLRClient import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_self_supervised.global_buffer import PrivacyPreservingBuffer
            print("   ✓ PrivacyPreservingBuffer imported")
        except Exception as e:
            print(f"   ✗ PrivacyPreservingBuffer import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_self_supervised.momentum_encoder import FederatedMomentumEncoder
            print("   ✓ FederatedMomentumEncoder imported")
        except Exception as e:
            print(f"   ✗ FederatedMomentumEncoder import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_self_supervised.contrastive_loss import NTXentLoss
            print("   ✓ NTXentLoss imported")
        except Exception as e:
            print(f"   ✗ NTXentLoss import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_self_supervised.fed_simclr import FedSimCLRCoordinator, ResNetEncoder
            print("   ✓ FedSimCLRCoordinator imported")
        except Exception as e:
            print(f"   ✗ FedSimCLRCoordinator import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing ResNetEncoder...")
            encoder = ResNetEncoder(input_channels=3, output_dim=128)
            dummy = torch.randn(2, 3, 32, 32)
            output = encoder(dummy)
            print(f"   ✓ ResNetEncoder: {output.shape}")
        except Exception as e:
            print(f"   ✗ ResNetEncoder failed: {e}")
        
        try:
            print("   Testing NTXentLoss...")
            loss_fn = NTXentLoss(temperature=0.5)
            z1 = torch.randn(8, 128)
            z2 = torch.randn(8, 128)
            loss = loss_fn(z1, z2)
            print(f"   ✓ NTXentLoss: {loss.item():.4f}")
        except Exception as e:
            print(f"   ✗ NTXentLoss failed: {e}")
        
        try:
            print("   Testing PrivacyPreservingBuffer...")
            buffer = PrivacyPreservingBuffer(max_size=1024, feature_dim=128)
            features = torch.randn(32, 128)
            buffer.add(features)
            sampled = buffer.sample(16)
            print(f"   ✓ Buffer: added 32 features, sampled {sampled.shape[0]}")
        except Exception as e:
            print(f"   ✗ PrivacyPreservingBuffer failed: {e}")
        
        try:
            print("   Testing SimCLRClient...")
            client_encoder = ResNetEncoder(input_channels=3, output_dim=128)
            client = SimCLRClient(client_id=0, encoder=client_encoder, projection_dim=128)
            print(f"   ✓ SimCLRClient created")
        except Exception as e:
            print(f"   ✗ SimCLRClient failed: {e}")
        
        try:
            print("   Testing FedSimCLRCoordinator...")
            coordinator = FedSimCLRCoordinator(num_clients=2, num_rounds=1, local_epochs=1)
            coordinator.setup()
            print(f"   ✓ Coordinator created with {len(coordinator.clients)} clients")
        except Exception as e:
            print(f"   ✗ FedSimCLRCoordinator failed: {e}")
        
        print("\n✓ Federated Self-Supervised Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Self-Supervised Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_causal_representation_demo():
    """Run federated causal representation learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED CAUSAL REPRESENTATION LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_causal_representation.causal_vae_client import CausalVAE
            print("   ✓ CausalVAE imported")
        except Exception as e:
            print(f"   ✗ CausalVAE import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.graph_aggregator import CausalGraphAggregator
            print("   ✓ CausalGraphAggregator imported")
        except Exception as e:
            print(f"   ✗ CausalGraphAggregator import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.invariance_constraint import MMDInvariance
            print("   ✓ MMDInvariance imported")
        except Exception as e:
            print(f"   ✗ MMDInvariance import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.counterfactual import CounterfactualReasoner
            print("   ✓ CounterfactualReasoner imported")
        except Exception as e:
            print(f"   ✗ CounterfactualReasoner import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing CausalVAE...")
            vae = CausalVAE(input_dim=8, num_causal_vars=5, num_noise_vars=3)
            x = torch.randn(4, 8)
            outputs = vae(x)
            print(f"   ✓ CausalVAE: reconstruction shape {outputs['reconstruction'].shape}")
        except Exception as e:
            print(f"   ✗ CausalVAE failed: {e}")
        
        try:
            print("   Testing DAG constraint...")
            dag_loss = vae.sem.dag_constraint()
            print(f"   ✓ DAG constraint: {dag_loss.item():.4f}")
        except Exception as e:
            print(f"   ✗ DAG constraint failed: {e}")
        
        try:
            print("   Testing CausalGraphAggregator...")
            import numpy as np
            aggregator = CausalGraphAggregator(num_vars=5)
            adj1 = np.random.rand(5, 5) > 0.7
            adj2 = np.random.rand(5, 5) > 0.7
            aggregator.receive_graph(0, adj1)
            aggregator.receive_graph(1, adj2)
            global_adj = aggregator.aggregate()
            print(f"   ✓ Graph aggregator: shape {global_adj.shape}")
        except Exception as e:
            print(f"   ✗ Graph aggregator failed: {e}")
        
        try:
            print("   Testing MMDInvariance...")
            mmd = MMDInvariance()
            rep1 = torch.randn(4, 5)
            rep2 = torch.randn(4, 5)
            loss = mmd([rep1, rep2])
            print(f"   ✓ MMD invariance: loss {loss.item():.4f}")
        except Exception as e:
            print(f"   ✗ MMD invariance failed: {e}")
        
        print("\n✓ Federated Causal Representation Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Causal Representation Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_linear_connectivity_demo():
    """Run federated linear mode connectivity demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED LINEAR MODE CONNECTIVITY")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_linear_connectivity.connectivity_metrics import (
                ConnectivityScore, InterpolationLossEvaluator
            )
            print("   ✓ Connectivity metrics imported")
        except Exception as e:
            print(f"   ✗ Connectivity metrics import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_linear_connectivity.federated_lmc import (
                FederatedLMC, LMCCoordinator, ClientModelSnapshot
            )
            print("   ✓ Federated LMC components imported")
        except Exception as e:
            print(f"   ✗ Federated LMC import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_linear_connectivity.visualization import (
                plot_interpolation_path, plot_connectivity_evolution
            )
            print("   ✓ Visualization tools imported")
        except Exception as e:
            print(f"   ✗ Visualization import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing ConnectivityScore...")
            scores = [0.9, 0.85, 0.7, 0.95]
            connectivity_score = ConnectivityScore(threshold=0.1)
            for s in scores:
                connectivity_score.add_score(s)
            avg_score = connectivity_score.get_average_score()
            print(f"   ✓ Average connectivity score: {avg_score:.4f}")
        except Exception as e:
            print(f"   ✗ ConnectivityScore failed: {e}")
        
        try:
            print("   Testing ClientModelSnapshot...")
            model = torch.nn.Linear(10, 5)
            state_dict = model.state_dict()
            snapshot = ClientModelSnapshot(
                client_id=0,
                round_idx=10,
                model_state_dict=state_dict,
                sample_count=1000
            )
            print(f"   ✓ Snapshot created: client {snapshot.client_id}, round {snapshot.round_idx}")
        except Exception as e:
            print(f"   ✗ ClientModelSnapshot failed: {e}")
        
        try:
            print("   Testing LMCCoordinator...")
            coordinator = LMCCoordinator(
                model_class=torch.nn.Linear,
                loss_fn=torch.nn.CrossEntropyLoss(),
                evaluation_rounds=[20, 50, 80],
                sample_ratio=0.25
            )
            print(f"   ✓ Coordinator created with rounds {coordinator.evaluation_rounds}")
        except Exception as e:
            print(f"   ✗ LMCCoordinator failed: {e}")
        
        try:
            print("   Testing interpolation evaluation...")
            model1 = torch.nn.Linear(10, 5)
            model2 = torch.nn.Linear(10, 5)
            evaluator = InterpolationLossEvaluator(torch.nn.Linear, num_points=5)
            losses = evaluator.evaluate_interpolation(
                model1, model2,
                [(torch.randn(2, 10), torch.randint(0, 5, (2,)))],
                torch.nn.CrossEntropyLoss(),
                device='cpu'
            )
            print(f"   ✓ Interpolation evaluated: {len(losses)} points")
        except Exception as e:
            print(f"   ✗ Interpolation evaluation failed: {e}")
        
        print("\n✓ Federated Linear Mode Connectivity demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Linear Mode Connectivity demo: {e}")
        import traceback
        traceback.print_exc()


def run_topology_aware_demo():
    """Run topology-aware federated learning demonstration."""
    print("\n" + "=" * 70)
    print("  TOPOLOGY-AWARE FEDERATED LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.topology_aware.topology_discovery import NetworkTopology
            print("   ✓ NetworkTopology imported")
        except Exception as e:
            print(f"   ✗ NetworkTopology import failed: {e}")
            return
        
        try:
            from hermes_unified.topology_aware.hierarchical_aggregator import (
                HierarchicalAggregator, AggregationTree
            )
            print("   ✓ HierarchicalAggregator imported")
        except Exception as e:
            print(f"   ✗ HierarchicalAggregator import failed: {e}")
            return
        
        try:
            from hermes_unified.topology_aware.topology_scheduler import TopologyScheduler
            print("   ✓ TopologyScheduler imported")
        except Exception as e:
            print(f"   ✗ TopologyScheduler import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing NetworkTopology...")
            topology = NetworkTopology('small_world')
            topology.generate_random_topology(20)
            info = topology.get_topology_info()
            print(f"   ✓ Topology: {info['num_nodes']} nodes, {info['num_edges']} edges")
        except Exception as e:
            print(f"   ✗ NetworkTopology failed: {e}")
        
        try:
            print("   Testing AggregationTree...")
            tree = AggregationTree()
            tree.add_node(-1)
            tree.add_node(0)
            tree.add_edge(-1, 0)
            print(f"   ✓ Tree: {tree.get_num_nodes()} nodes")
        except Exception as e:
            print(f"   ✗ AggregationTree failed: {e}")
        
        try:
            print("   Testing TopologyScheduler...")
            scheduler = TopologyScheduler(topology, server_id=-1)
            scheduler.set_strategy('min_cost')
            selected = scheduler.select_participants(list(range(20)), 5, 0)
            print(f"   ✓ Selected {len(selected)} clients")
        except Exception as e:
            print(f"   ✗ TopologyScheduler failed: {e}")
        
        print("\n✓ Topology-Aware Federated Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Topology-Aware FL demo: {e}")
        import traceback
        traceback.print_exc()


def run_online_learning_demo():
    """Run federated online learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED ONLINE LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.federated_online.drift_detector import (
                CUSUMDetector, ADWINDetector
            )
            print("   ✓ Drift detectors imported")
        except Exception as e:
            print(f"   ✗ Drift detectors import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.online_client import OnlineClient
            print("   ✓ OnlineClient imported")
        except Exception as e:
            print(f"   ✗ OnlineClient import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.online_server import OnlineServer
            print("   ✓ OnlineServer imported")
        except Exception as e:
            print(f"   ✗ OnlineServer import failed: {e}")
            return
        
        try:
            from hermes_unified.federated_online.fomaml import FOMAMLClient, FOMAMLServer
            print("   ✓ FOMAML components imported")
        except Exception as e:
            print(f"   ✗ FOMAML import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing DriftDetector...")
            detector = CUSUMDetector(threshold=5.0)
            for i in range(100):
                detector.update(0.1)
            print("   ✓ CUSUM detector initialized")
        except Exception as e:
            print(f"   ✗ DriftDetector failed: {e}")
        
        try:
            print("   Testing OnlineClient...")
            model = torch.nn.Linear(10, 1)
            client = OnlineClient(0, model, torch.nn.MSELoss())
            print(f"   ✓ OnlineClient created")
        except Exception as e:
            print(f"   ✗ OnlineClient failed: {e}")
        
        try:
            print("   Testing OnlineServer...")
            server = OnlineServer(torch.nn.Linear(10, 1))
            print(f"   ✓ OnlineServer created")
        except Exception as e:
            print(f"   ✗ OnlineServer failed: {e}")
        
        print("\n✓ Federated Online Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Online Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_shapley_demo():
    """Run federated Shapley value learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED SHAPLEY VALUE LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.shapley.shapley_estimator import (
                MonteCarloShapleyEstimator, ShapleyValue
            )
            print("   ✓ Shapley estimators imported")
        except Exception as e:
            print(f"   ✗ Shapley estimators import failed: {e}")
            return
        
        try:
            from hermes_unified.shapley.incentive_mechanism import (
                TokenDistributor, ReputationSystem
            )
            print("   ✓ Incentive mechanisms imported")
        except Exception as e:
            print(f"   ✗ Incentive mechanisms import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing MonteCarloShapleyEstimator...")
            estimator = MonteCarloShapleyEstimator(num_clients=5, num_samples=100)
            
            def valuation_fn(subset):
                return len(subset) * 0.2
            
            values = estimator.estimate([0, 1, 2, 3, 4], valuation_fn)
            print(f"   ✓ Estimated {len(values)} Shapley values")
        except Exception as e:
            print(f"   ✗ MonteCarloShapleyEstimator failed: {e}")
        
        try:
            print("   Testing TokenDistributor...")
            distributor = TokenDistributor(total_tokens_per_round=100)
            distributor.distribute({0: 0.5, 1: 0.3, 2: 0.2})
            print(f"   ✓ Tokens distributed: {distributor.get_total_distributed()}")
        except Exception as e:
            print(f"   ✗ TokenDistributor failed: {e}")
        
        try:
            print("   Testing ReputationSystem...")
            reputation = ReputationSystem()
            reputation.update_reputation(0, 0.8)
            print(f"   ✓ Reputation updated: {reputation.get_reputation(0):.4f}")
        except Exception as e:
            print(f"   ✗ ReputationSystem failed: {e}")
        
        print("\n✓ Federated Shapley Value Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in Shapley demo: {e}")
        import traceback
        traceback.print_exc()


def run_fednas_demo():
    """Run federated neural architecture search demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED NEURAL ARCHITECTURE SEARCH")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_nas.supernet import SuperNet, DartsSearchSpace
            print("   ✓ SuperNet and SearchSpace imported")
        except Exception as e:
            print(f"   ✗ SuperNet import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_nas.subnet_extractor import (
                SubnetExtractor, ResourceEvaluator
            )
            print("   ✓ SubnetExtractor imported")
        except Exception as e:
            print(f"   ✗ SubnetExtractor import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_nas.local_search import (
                EvolutionarySearch, SearchFactory
            )
            print("   ✓ Local search imported")
        except Exception as e:
            print(f"   ✗ Local search import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing SuperNet...")
            search_space = DartsSearchSpace()
            supernet = SuperNet(search_space, num_classes=10)
            dummy_input = torch.randn(2, 3, 32, 32)
            output = supernet(dummy_input)
            print(f"   ✓ SuperNet created, output shape: {output.shape}")
        except Exception as e:
            print(f"   ✗ SuperNet failed: {e}")
        
        try:
            print("   Testing ResourceEvaluator...")
            evaluator = ResourceEvaluator()
            resources = evaluator.evaluate(supernet, (3, 32, 32))
            print(f"   ✓ Resource evaluation: FLOPs={resources['flops']:.2e}")
        except Exception as e:
            print(f"   ✗ ResourceEvaluator failed: {e}")
        
        print("\n✓ Federated Neural Architecture Search demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in FedNAS demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated Learning Multi-Direction Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--direction', type=str, default='all',
        choices=['peft', 'transfer', 'quantum', 'multimodal', 'selfsupervised', 'causal', 'linearconnectivity', 'topology', 'online', 'shapley', 'fednas', 'all'],
        help='Which direction to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  HERMES COSMOS - MULTI-DIRECTION FEDERATED LEARNING")
    print("=" * 70)
    
    if args.direction == 'peft' or args.direction == 'all':
        run_fed_peft_demo()
    
    if args.direction == 'transfer' or args.direction == 'all':
        run_transfer_learning_demo()
    
    if args.direction == 'quantum' or args.direction == 'all':
        run_quantum_learning_demo()
    
    if args.direction == 'multimodal' or args.direction == 'all':
        run_multimodal_demo()
    
    if args.direction == 'selfsupervised' or args.direction == 'all':
        run_self_supervised_demo()
    
    if args.direction == 'causal' or args.direction == 'all':
        run_causal_representation_demo()
    
    if args.direction == 'linearconnectivity' or args.direction == 'all':
        run_linear_connectivity_demo()
    
    if args.direction == 'topology' or args.direction == 'all':
        run_topology_aware_demo()
    
    if args.direction == 'online' or args.direction == 'all':
        run_online_learning_demo()
    
    if args.direction == 'shapley' or args.direction == 'all':
        run_shapley_demo()
    
    if args.direction == 'fednas' or args.direction == 'all':
        run_fednas_demo()
    
    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()