"""
LSTM with Attention Implementation Analysis
==========================================

This document explains the improvements made to the LSTM implementation to ensure
proper interaction between LSTM blocks and attention mechanisms.

## Previous Implementation Issues:

1. **No Layer-wise Attention**:

   - Had only a single attention mechanism applied AFTER all LSTM layers
   - No interaction between attention and individual LSTM layers

2. **Simple Attention**:

   - Used only a basic linear layer for attention weights
   - No multi-head attention mechanism
   - No sophisticated attention patterns

3. **Architecture Mismatch**:
   - Was more of an encoder-decoder with global attention
   - Not truly comparable to transformer architecture

## New Implementation: LSTMAttentionBlock

### Architecture Overview:

```
Input → LSTMAttentionBlock₁ → LSTMAttentionBlock₂ → ... → LSTMAttentionBlockₙ → Output
         ↓                    ↓                         ↓
    [LSTM + Attention]   [LSTM + Attention]       [LSTM + Attention]
```

### Each LSTMAttentionBlock contains:

1. **LSTM Layer**:

   - Single LSTM cell with hidden_size dimensions
   - Processes sequential information

2. **Multi-Head Self-Attention**:

   - Query, Key, Value projections
   - Scaled dot-product attention
   - Multiple attention heads for different representation subspaces
   - Attention mask support for padding tokens

3. **Residual Connections & Layer Normalization**:
   - Skip connections around both LSTM and attention
   - Layer normalization for training stability
   - Similar to transformer architecture

### Forward Pass in Each Block:

```python
# 1. LSTM processing with residual
residual = x
lstm_out, _ = self.lstm(x)
x = self.ln1(lstm_out + residual)  # Layer norm + residual

# 2. Multi-head attention with residual
residual = x
q, k, v = self.query(x), self.key(x), self.value(x)
attention_out = multi_head_attention(q, k, v, mask)
x = self.ln2(attention_out + residual)  # Layer norm + residual
```

## Key Improvements:

### 1. **Per-Layer Attention Interaction**:

- Each LSTM layer is followed by an attention mechanism
- Attention can refine and redistribute information processed by LSTM
- Creates a hybrid architecture combining sequential and parallel processing

### 2. **Multi-Head Attention**:

- Uses multiple attention heads (configurable, default 4)
- Each head can focus on different aspects of the sequence
- More expressive than single attention mechanism

### 3. **Proper Residual Architecture**:

- Skip connections around both LSTM and attention components
- Helps with gradient flow and training stability
- Enables deeper architectures

### 4. **Transformer-like Design**:

- Layer normalization placement similar to transformers
- Proper initialization schemes
- Makes the architecture more comparable to GPT model

## Architecture Comparison:

### Old LSTM:

```
Embedding → Multi-layer LSTM → Global Attention → Decoder LSTM → Classification
```

### New LSTM with Attention:

```
Embedding → [LSTM + Attention]₁ → [LSTM + Attention]₂ → ... → Classification
```

### Transformer (GPT):

```
Embedding → [Multi-Head Attention + MLP]₁ → [Multi-Head Attention + MLP]₂ → ... → Classification
```

## Benefits of the New Architecture:

1. **Fair Comparison**: Now both models use attention mechanisms at each layer
2. **Better Representation**: LSTM handles sequential dependencies, attention handles global dependencies
3. **Hybrid Strengths**: Combines inductive biases of RNNs with attention mechanisms
4. **Scalability**: Can stack multiple blocks like transformers
5. **Training Stability**: Proper residual connections and normalization

## Parameter Analysis:

- **LSTM with Attention**: ~6.83M parameters
- **GPT Transformer**: ~6.90M parameters

The parameter counts are now very similar, making the comparison more fair and meaningful.

## Usage:

The new implementation maintains the same interface:

```python
lstm_config.n_head = 4  # Number of attention heads
model = LSTMClassifier(lstm_config)
logits, loss = model(input_ids, attention_mask, labels)
```

This creates a proper hybrid architecture where each LSTM block truly interacts with attention,
making it a more robust baseline for comparison with transformer models.
"""
