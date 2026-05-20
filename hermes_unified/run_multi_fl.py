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
        lora = AdaptiveLoRAModule(model, rank=4)
        print(f"   Trainable params: {lora.count_trainable_parameters():,}")

        print("\n[OK] FedPEFT demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in FedPEFT demo: {e}")
        import traceback
        traceback.print_exc()


def run_transfer_learning_demo():
    """Run federated transfer learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED TRANSFER LEARNING")
    print("=" * 70)

    try:
        import torch

        print("\n1. Checking torch installation...")
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.transfer_learning.domain_adaptation import (
                MMDLoss, DomainDiscriminator, DomainAdversarialAdaptor
            )
            print("   [OK] Domain adaptation imported")
        except Exception as e:
            print(f"   [FAIL] Domain adaptation import failed: {e}")
            return

        try:
            from hermes_unified.transfer_learning.knowledge_distillation import (
                FedDistillationClient, FedTransferServer
            )
            print("   [OK] Knowledge distillation imported")
        except Exception as e:
            print(f"   [FAIL] Knowledge distillation import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing MMDLoss...")
            loss_fn = MMDLoss()
            print("   [OK] MMDLoss created")
        except Exception as e:
            print(f"   [FAIL] MMDLoss failed: {e}")

        try:
            print("   Testing DomainDiscriminator...")
            discriminator = DomainDiscriminator(input_dim=128, hidden_dim=64)
            print("   [OK] DomainDiscriminator created")
        except Exception as e:
            print(f"   [FAIL] DomainDiscriminator failed: {e}")

        print("\n[OK] Federated Transfer Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Transfer Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_quantum_learning_demo():
    """Run federated quantum learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED QUANTUM LEARNING")
    print("=" * 70)

    try:
        import torch

        print("\n1. Checking torch installation...")
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.q_fl.fed_qnn import (
                QuantumCircuit, QuantumClassifier, FedQNNClient, FedQNNServer
            )
            print("   [OK] Quantum components imported")
        except Exception as e:
            print(f"   [FAIL] Quantum components import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing QuantumCircuit...")
            qc = QuantumCircuit(num_qubits=2, num_layers=2)
            print("   [OK] QuantumCircuit created")
        except Exception as e:
            print(f"   [FAIL] QuantumCircuit failed: {e}")

        try:
            print("   Testing QuantumClassifier...")
            classifier = QuantumClassifier(num_qubits=2, num_classes=2)
            print("   [OK] QuantumClassifier created")
        except Exception as e:
            print(f"   [FAIL] QuantumClassifier failed: {e}")

        print("\n[OK] Federated Quantum Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Quantum Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_multimodal_demo():
    """Run federated multimodal learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED MULTIMODAL LEARNING")
    print("=" * 70)

    try:
        import torch

        print("\n1. Checking torch installation...")
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.fed_multimodal.multimodal_client import (
                MultimodalFedClient, ImageEncoder, TextEncoder, TabularEncoder
            )
            print("   [OK] Multimodal client imported")
        except Exception as e:
            print(f"   [FAIL] Multimodal client import failed: {e}")
            return

        try:
            from hermes_unified.fed_multimodal.multimodal_server import (
                MultimodalFedServer, DifferentialPrivacyBudget
            )
            print("   [OK] Multimodal server imported")
        except Exception as e:
            print(f"   [FAIL] Multimodal server import failed: {e}")
            return

        try:
            from hermes_unified.fed_multimodal.modality_aligner import (
                CLIPStyleAligner, ContrastiveAligner
            )
            print("   [OK] Modality aligner imported")
        except Exception as e:
            print(f"   [FAIL] Modality aligner import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing ImageEncoder...")
            img_encoder = ImageEncoder(embedding_dim=64)
            print("   [OK] ImageEncoder created")
        except Exception as e:
            print(f"   [FAIL] ImageEncoder failed: {e}")

        try:
            print("   Testing TextEncoder...")
            txt_encoder = TextEncoder(embedding_dim=64)
            print("   [OK] TextEncoder created")
        except Exception as e:
            print(f"   [FAIL] TextEncoder failed: {e}")

        try:
            print("   Testing CLIPStyleAligner...")
            aligner = CLIPStyleAligner(embedding_dim=64)
            print("   [OK] CLIPStyleAligner created")
        except Exception as e:
            print(f"   [FAIL] CLIPStyleAligner failed: {e}")

        print("\n[OK] Federated Multimodal Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Multimodal Learning demo: {e}")
        import traceback
        traceback.print_exc()


def run_self_supervised_demo():
    """Run federated self-supervised learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED SELF-SUPERVISED LEARNING")
    print("=" * 70)

    try:
        import torch

        print("\n1. Checking torch installation...")
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.fed_self_supervised.contrastive_loss import (
                NTXentLoss, MoCoLoss, BarlowTwinsLoss
            )
            print("   [OK] Contrastive loss imported")
        except Exception as e:
            print(f"   [FAIL] Contrastive loss import failed: {e}")
            return

        try:
            from hermes_unified.fed_self_supervised.contrastive_client import (
                ContrastiveClient
            )
            print("   [OK] Contrastive client imported")
        except Exception as e:
            print(f"   [FAIL] Contrastive client import failed: {e}")
            return

        try:
            from hermes_unified.fed_self_supervised.global_buffer import (
                PrivacyPreservingBuffer
            )
            print("   [OK] Global buffer imported")
        except Exception as e:
            print(f"   [FAIL] Global buffer import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing NTXentLoss...")
            loss_fn = NTXentLoss(temperature=0.5)
            print("   [OK] NTXentLoss created")
        except Exception as e:
            print(f"   [FAIL] NTXentLoss failed: {e}")

        try:
            print("   Testing PrivacyPreservingBuffer...")
            buffer = PrivacyPreservingBuffer(buffer_size=1000)
            print("   [OK] PrivacyPreservingBuffer created")
        except Exception as e:
            print(f"   [FAIL] PrivacyPreservingBuffer failed: {e}")

        print("\n[OK] Federated Self-Supervised Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Self-Supervised Learning demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.fed_causal_representation.causal_vae_client import (
                CausalVAE, StructuralEquationModel
            )
            print("   [OK] CausalVAE imported")
        except Exception as e:
            print(f"   [FAIL] CausalVAE import failed: {e}")
            return

        try:
            from hermes_unified.fed_causal_representation.graph_aggregator import (
                CausalGraphAggregator
            )
            print("   [OK] Causal graph aggregator imported")
        except Exception as e:
            print(f"   [FAIL] Causal graph aggregator import failed: {e}")
            return

        try:
            from hermes_unified.fed_causal_representation.invariance_constraint import (
                MMDInvariance, IRMInvariance
            )
            print("   [OK] Invariance constraints imported")
        except Exception as e:
            print(f"   [FAIL] Invariance constraints import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing CausalVAE...")
            causal_vae = CausalVAE(input_dim=10, latent_dim=5, num_causal_vars=3)
            print("   [OK] CausalVAE created")
        except Exception as e:
            print(f"   [FAIL] CausalVAE failed: {e}")

        try:
            print("   Testing MMDInvariance...")
            mmd_inv = MMDInvariance()
            print("   [OK] MMDInvariance created")
        except Exception as e:
            print(f"   [FAIL] MMD invariance failed: {e}")

        print("\n[OK] Federated Causal Representation Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Causal Representation Learning demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.fed_linear_connectivity.connectivity_metrics import (
                ConnectivityScore, InterpolationLossEvaluator
            )
            print("   [OK] Connectivity metrics imported")
        except Exception as e:
            print(f"   [FAIL] Connectivity metrics import failed: {e}")
            return

        try:
            from hermes_unified.fed_linear_connectivity.federated_lmc import (
                FederatedLMC, LMCCoordinator, ClientModelSnapshot
            )
            print("   [OK] Federated LMC components imported")
        except Exception as e:
            print(f"   [FAIL] Federated LMC import failed: {e}")
            return

        try:
            from hermes_unified.fed_linear_connectivity.visualization import (
                plot_interpolation_path, plot_connectivity_evolution
            )
            print("   [OK] Visualization tools imported")
        except Exception as e:
            print(f"   [FAIL] Visualization import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing ConnectivityScore...")
            scores = [0.9, 0.85, 0.7, 0.95]
            connectivity_score = ConnectivityScore(threshold=0.1)
            for s in scores:
                connectivity_score.update(s)
            print("   [OK] ConnectivityScore created")
        except Exception as e:
            print(f"   [FAIL] ConnectivityScore failed: {e}")

        try:
            print("   Testing LMCCoordinator...")
            coordinator = LMCCoordinator(num_clients=5)
            print("   [OK] LMCCoordinator created")
        except Exception as e:
            print(f"   [FAIL] LMCCoordinator failed: {e}")

        print("\n[OK] Federated Linear Mode Connectivity demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Linear Mode Connectivity demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.topology_aware.topology_discovery import (
                NetworkTopology, TopologyDiscovery
            )
            print("   [OK] Topology discovery imported")
        except Exception as e:
            print(f"   [FAIL] Topology discovery import failed: {e}")
            return

        try:
            from hermes_unified.topology_aware.hierarchical_aggregator import (
                AggregationTree, HierarchicalAggregator
            )
            print("   [OK] Hierarchical aggregator imported")
        except Exception as e:
            print(f"   [FAIL] Hierarchical aggregator import failed: {e}")
            return

        try:
            from hermes_unified.topology_aware.topology_scheduler import (
                TopologyScheduler
            )
            print("   [OK] Topology scheduler imported")
        except Exception as e:
            print(f"   [FAIL] Topology scheduler import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing NetworkTopology...")
            topology = NetworkTopology(num_nodes=10)
            print("   [OK] NetworkTopology created")
        except Exception as e:
            print(f"   [FAIL] NetworkTopology failed: {e}")

        try:
            print("   Testing HierarchicalAggregator...")
            aggregator = HierarchicalAggregator(num_levels=3)
            print("   [OK] HierarchicalAggregator created")
        except Exception as e:
            print(f"   [FAIL] HierarchicalAggregator failed: {e}")

        print("\n[OK] Topology-Aware Federated Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Topology-Aware FL demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.federated_online.drift_detector import (
                CUSUMDetector, ADWINDetector
            )
            print("   [OK] Drift detectors imported")
        except Exception as e:
            print(f"   [FAIL] Drift detectors import failed: {e}")
            return

        try:
            from hermes_unified.federated_online.online_client import OnlineClient
            print("   [OK] OnlineClient imported")
        except Exception as e:
            print(f"   [FAIL] OnlineClient import failed: {e}")
            return

        try:
            from hermes_unified.federated_online.online_server import OnlineServer
            print("   [OK] OnlineServer imported")
        except Exception as e:
            print(f"   [FAIL] OnlineServer import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing DriftDetector...")
            detector = CUSUMDetector(threshold=5.0)
            for i in range(100):
                detector.update(0.1)
            print("   [OK] CUSUM detector initialized")
        except Exception as e:
            print(f"   [FAIL] DriftDetector failed: {e}")

        try:
            print("   Testing OnlineClient...")
            model = torch.nn.Linear(10, 1)
            client = OnlineClient(0, model, torch.nn.MSELoss())
            print("   [OK] OnlineClient created")
        except Exception as e:
            print(f"   [FAIL] OnlineClient failed: {e}")

        try:
            print("   Testing OnlineServer...")
            server = OnlineServer(torch.nn.Linear(10, 1))
            print("   [OK] OnlineServer created")
        except Exception as e:
            print(f"   [FAIL] OnlineServer failed: {e}")

        print("\n[OK] Federated Online Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Online Learning demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.shapley.shapley_estimator import (
                MonteCarloShapleyEstimator, ShapleyValue
            )
            print("   [OK] Shapley estimators imported")
        except Exception as e:
            print(f"   [FAIL] Shapley estimators import failed: {e}")
            return

        try:
            from hermes_unified.shapley.incentive_mechanism import (
                TokenDistributor, ReputationSystem
            )
            print("   [OK] Incentive mechanisms imported")
        except Exception as e:
            print(f"   [FAIL] Incentive mechanisms import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        try:
            print("   Testing MonteCarloShapleyEstimator...")
            estimator = MonteCarloShapleyEstimator(num_clients=5, num_samples=100)

            def valuation_fn(subset):
                return len(subset) * 0.2

            values = estimator.estimate([0, 1, 2, 3, 4], valuation_fn)
            print(f"   [OK] Estimated {len(values)} Shapley values")
        except Exception as e:
            print(f"   [FAIL] MonteCarloShapleyEstimator failed: {e}")

        try:
            print("   Testing TokenDistributor...")
            distributor = TokenDistributor(total_tokens_per_round=100)
            distributor.distribute({0: 0.5, 1: 0.3, 2: 0.2})
            print(f"   [OK] Tokens distributed: {distributor.get_total_distributed()}")
        except Exception as e:
            print(f"   [FAIL] TokenDistributor failed: {e}")

        try:
            print("   Testing ReputationSystem...")
            reputation = ReputationSystem()
            reputation.update_reputation(0, 0.8)
            print(f"   [OK] Reputation updated: {reputation.get_reputation(0):.4f}")
        except Exception as e:
            print(f"   [FAIL] ReputationSystem failed: {e}")

        print("\n[OK] Federated Shapley Value Learning demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in Shapley demo: {e}")
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
        print(f"   [OK] torch version: {torch.__version__}")

        print("\n2. Testing Core Components Import:")

        try:
            from hermes_unified.fed_nas.supernet import SuperNet
            print("   [OK] SuperNet imported")
        except Exception as e:
            print(f"   [FAIL] SuperNet import failed: {e}")
            return

        try:
            from hermes_unified.fed_nas.subnet_extractor import (
                SubnetExtractor, ResourceEvaluator
            )
            print("   [OK] SubnetExtractor imported")
        except Exception as e:
            print(f"   [FAIL] SubnetExtractor import failed: {e}")
            return

        try:
            from hermes_unified.fed_nas.local_search import (
                EvolutionarySearch, SearchFactory
            )
            print("   [OK] Local search imported")
        except Exception as e:
            print(f"   [FAIL] Local search import failed: {e}")
            return

        print("\n3. Testing Component Functionality:")

        supernet = None
        try:
            print("   Testing SuperNet...")
            supernet = SuperNet(num_classes=10, num_cells=6, channels=32)
            dummy_input = torch.randn(2, 3, 32, 32)
            output = supernet(dummy_input)
            print(f"   [OK] SuperNet created, output shape: {output.shape}")
        except Exception as e:
            print(f"   [FAIL] SuperNet failed: {e}")

        if supernet is not None:
            try:
                print("   Testing ResourceEvaluator...")
                evaluator = ResourceEvaluator()
                resources = evaluator.evaluate(supernet, (3, 32, 32))
                print(f"   [OK] Resource evaluation: FLOPs={resources['flops']:.2e}")
            except Exception as e:
                print(f"   [FAIL] ResourceEvaluator failed: {e}")
        else:
            print("   [SKIP] ResourceEvaluator skipped (SuperNet not available)")

        print("\n[OK] Federated Neural Architecture Search demo completed!")

    except ImportError as e:
        print(f"[WARN] Import error: {e}")
    except Exception as e:
        print(f"[WARN] Error in FedNAS demo: {e}")
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