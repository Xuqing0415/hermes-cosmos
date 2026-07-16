# LLM Federated Fine-Tuning Module

This module implements federated learning for Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) to minimize communication overhead.

## Features

- **LoRA Integration**: Train only low-rank matrices, reducing communication from GBs to MBs
- **Non-IID Data Support**: Simulate realistic data distribution across clients
- **Multiple Aggregation Methods**: FedAvg, Krum, Median
- **Centralized Comparison**: Compare federated performance with centralized training
- **Perplexity Evaluation**: Track model quality throughout training

## Requirements

```bash
pip install torch transformers peft datasets accelerate evaluate
```

## Quick Start

### Basic Usage

```python
from llm_finetune.run_llm_finetune import run_federated_finetune, run_centralized_finetune

# Run federated fine-tuning
fed_results = run_federated_finetune(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_clients=5,
    num_rounds=10,
    clients_per_round=3,
    local_epochs=1,
    non_iid_alpha=0.5
)

# Run centralized fine-tuning for comparison
central_results = run_centralized_finetune(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs=3
)
```

### Running the Experiment Script

```bash
cd hermes_unified/llm_finetune
python run_llm_finetune.py
```

## Module Structure

```
llm_finetune/
 llm_model.py          # Model loading and LoRA wrapping
 non_iid_data.py       # Non-IID text data generation
 llm_client.py         # Federated client implementation
 llm_server.py         # Federated server and coordinator
 run_llm_finetune.py   # Main experiment runner
 README.md             # This file
```

## Key Components

### 1. LLM Model Loading (`llm_model.py`)

- `load_llm()`: Load pre-trained LLM from HuggingFace
- `wrap_with_lora()`: Wrap model with LoRA adapter
- `get_lora_weights()`: Extract LoRA weights for communication
- `calculate_lora_size()`: Calculate communication size

### 2. Non-IID Data (`non_iid_data.py`)

- `create_non_iid_text_data()`: Create Non-IID data using Dirichlet distribution
- `create_synthetic_non_iid_data()`: Generate synthetic topic-specific data
- `tokenize_dataset()`: Prepare data for training

### 3. Federated Client (`llm_client.py`)

- `LLMFederatedClient`: Client with local LoRA training
- `local_train()`: Perform local training and return weight updates
- `evaluate()`: Evaluate local model

### 4. Federated Server (`llm_server.py`)

- `LLMFederatedServer`: Aggregates client updates
- `aggregate()`: Supports FedAvg, Krum, and Median aggregation
- `evaluate()`: Compute perplexity on test data

### 5. Experiment Runner (`run_llm_finetune.py`)

- `run_federated_finetune()`: Run federated training
- `run_centralized_finetune()`: Run centralized training
- `print_comparison_summary()`: Compare results
- `plot_results()`: Generate visualization

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model name | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `num_clients` | Number of federated clients | 5 |
| `num_rounds` | Number of federated rounds | 10 |
| `clients_per_round` | Clients selected per round | 3 |
| `local_epochs` | Local training epochs | 1 |
| `non_iid_alpha` | Non-IIDness (0=most non-IID) | 0.5 |
| `use_quantization` | Enable 4-bit quantization | False |
| `aggregation_method` | fedavg/krum/median | fedavg |

## Expected Output

```
=== LLM Federated Fine-Tuning Experiment ===
Comparing Federated vs Centralized LoRA Fine-Tuning

Running Federated LLM Fine-Tuning
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Clients: 5, Rounds: 10
Non-IID Alpha: 0.5

Round 1 Perplexity: 12.34
Round 2 Perplexity: 10.23
...

Running Centralized LLM Fine-Tuning
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Epochs: 3, Batch Size: 4

Epoch 1 Loss: 2.345, Perplexity: 10.34
...

COMPARISON SUMMARY
1. PERPLEXITY
Federated Final Perplexity: 8.56
Centralized Final Perplexity: 8.12
Difference: 0.44
 Federated performance is within 5% of centralized!

2. COMMUNICATION
Total Communication (Federated): 2.34 MB
Full Model Size (Estimated): ~28 GB (7B model)
LoRA Size Reduction: 0.008% of full model
```

## Notes

1. **Model Selection**: TinyLlama is used for testing due to its small size. For better performance, consider larger models like LLaMA-2 7B.

2. **Quantization**: Enable `use_quantization=True` for memory-efficient training on GPUs with limited VRAM.

3. **Non-IID Alpha**: Lower values (0.1-0.3) create more challenging non-IID scenarios.

4. **Communication Savings**: LoRA reduces communication by ~1000x compared to full model updates.

## References

- LoRA: https://arxiv.org/abs/2106.09685
- Federated Learning: https://arxiv.org/abs/1602.05629
