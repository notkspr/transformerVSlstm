#!/usr/bin/env python3
"""
Analyze parameter differences between LSTM and GPT models
"""
import torch
from model import LSTMClassifier, GPT
from bpe import BPETokenizer

# Create tokenizer
tokenizer = BPETokenizer()

# Common configuration
vocab_size = tokenizer.vocab_size + 2  # 50257 + 2 = 50259
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_heads = 4
max_len = 512

print(f"Configuration:")
print(f"  Vocab size: {vocab_size:,}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Num layers: {num_layers}")
print(f"  Max len: {max_len}")

# LSTM Configuration
lstm_config = LSTMClassifier.get_default_config()
lstm_config.vocab_size = vocab_size
lstm_config.block_size = hidden_dim  # Note: This is set to hidden_dim (256), not max_len!
lstm_config.n_embd = embedding_dim
lstm_config.n_layer = num_layers
lstm_config.n_head = num_heads
lstm_config.num_classes = 2

# GPT Configuration  
gpt_config = GPT.get_default_config()
gpt_config.model_type = None
gpt_config.vocab_size = vocab_size
gpt_config.block_size = max_len  # This is set to max_len (512)
gpt_config.n_embd = embedding_dim
gpt_config.n_head = num_heads
gpt_config.n_layer = num_layers
gpt_config.num_classes = 2

# Create models
lstm_model = LSTMClassifier(lstm_config)
gpt_model = GPT(gpt_config)

def analyze_model_parameters(model, name):
    print(f"\n{name} PARAMETER BREAKDOWN:")
    print("-" * 50)
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{param_name:35s}: {param_count:>10,} {list(param.shape)}")
    print("-" * 50)
    print(f"{'TOTAL':35s}: {total_params:>10,}")
    return total_params

lstm_params = analyze_model_parameters(lstm_model, "LSTM")
gpt_params = analyze_model_parameters(gpt_model, "GPT")

print(f"\n" + "="*60)
print("PARAMETER COMPARISON SUMMARY")
print("="*60)
print(f"LSTM total parameters: {lstm_params:>10,}")
print(f"GPT total parameters:  {gpt_params:>10,}")
print(f"Difference:            {lstm_params - gpt_params:>10,}")
print(f"LSTM/GPT ratio:        {lstm_params / gpt_params:.2f}x")

print(f"\n" + "="*60)
print("ROOT CAUSE ANALYSIS")
print("="*60)

# Main differences
lstm_embedding = 6_433_152
gpt_embedding = 6_433_152  
gpt_positional = 65_536
lstm_extra_lm_head = 6_433_152

print(f"1. EMBEDDING LAYERS:")
print(f"   Both models: {lstm_embedding:,} parameters (vocab_size × embedding_dim)")

print(f"\n2. CRITICAL ISSUE - LSTM has DUPLICATE embedding weights:")
print(f"   - embedding.weight: {lstm_embedding:,} parameters")
print(f"   - lm_head.weight:   {lstm_extra_lm_head:,} parameters") 
print(f"   Total redundancy:   {lstm_extra_lm_head:,} parameters (48.5% of LSTM!)")

print(f"\n3. POSITIONAL EMBEDDINGS:")
print(f"   - GPT has positional embeddings: {gpt_positional:,} parameters (block_size × embedding_dim)")
print(f"   - LSTM doesn't need positional embeddings (sequential processing)")

print(f"\n4. CORE ARCHITECTURE:")
lstm_core = lstm_params - lstm_embedding - lstm_extra_lm_head
gpt_core = gpt_params - gpt_embedding - gpt_positional
print(f"   - LSTM core (without duplicate embeddings): {lstm_core:,}")
print(f"   - GPT core (without embeddings):           {gpt_core:,}")
print(f"   - Core difference:                         {lstm_core - gpt_core:,}")

print(f"\n" + "="*60)
print("SOLUTION")
print("="*60)
print("The LSTM has duplicate embedding weights!")
print("- embedding.weight (input) and lm_head.weight (output) are both full vocab_size × embedding_dim")
print("- For fine-tuning, lm_head should be removed or shared with embedding")
print("- This explains the 13.26M vs 6.90M difference (almost exactly 2x embedding size!)")

# Check if they're actually the same weights or separate
print(f"\nWEIGHT SHARING CHECK:")
print(f"embedding.weight shape: {lstm_model.embedding.weight.shape}")
print(f"lm_head.weight shape:   {lstm_model.lm_head.weight.shape}")
print(f"Are they the same object? {lstm_model.embedding.weight is lstm_model.lm_head.weight}")