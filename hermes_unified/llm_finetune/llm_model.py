"""
LLM Model Loading and LoRA Wrapping Module

Provides utilities for loading LLMs and wrapping them with LoRA adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import os


def load_llm(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
             use_quantization: bool = False,
             device_map: str = "auto"):
    """
    Load a pre-trained LLM and tokenizer.
    
    Args:
        model_name: HuggingFace model name or path
        use_quantization: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    print(f"Loading model: {model_name}")
    
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if not use_quantization else torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully. Device: {model.device}")
    print(f"Number of parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def wrap_with_lora(model, r: int = 8, lora_alpha: int = 32, 
                   target_modules: list = None, lora_dropout: float = 0.1):
    """
    Wrap a model with LoRA adapter.
    
    Args:
        model: Base model to wrap
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: List of modules to apply LoRA to
        lora_dropout: Dropout rate for LoRA layers
        
    Returns:
        model: Model with LoRA adapter
        config: LoRA configuration
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, lora_config


def get_lora_weights(model) -> dict:
    """Extract LoRA weights from a PEFT model."""
    if isinstance(model, PeftModel):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k.lower()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if any(m in k.lower() for m in ['lora', 'adapter'])}


def set_lora_weights(model, weights: dict):
    """Set LoRA weights on a PEFT model."""
    current_state = model.state_dict()
    for k, v in weights.items():
        if k in current_state:
            current_state[k].data.copy_(v.to(model.device))
    model.load_state_dict(current_state, strict=False)


def merge_lora_weights(model, save_path: str = None):
    """Merge LoRA weights into base model."""
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    
    if save_path:
        model.save_pretrained(save_path)
    
    return model


def count_lora_parameters(model) -> int:
    """Count trainable LoRA parameters."""
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params


def calculate_lora_size(model) -> float:
    """Calculate the size of LoRA weights in MB."""
    weights = get_lora_weights(model)
    total_bytes = sum(p.numel() * p.element_size() for p in weights.values())
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    # Test loading
    model, tokenizer = load_llm(use_quantization=False)
    model, lora_config = wrap_with_lora(model)
    
    lora_size = calculate_lora_size(model)
    print(f"\nLoRA weights size: {lora_size:.2f} MB")
    
    # Test weight extraction
    weights = get_lora_weights(model)
    print(f"Number of LoRA weight tensors: {len(weights)}")
