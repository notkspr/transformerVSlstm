"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class UnifiedMultiHeadAttention(nn.Module):
    """
    Unified multi-head self-attention mechanism used by both LSTM and Transformer models.
    Supports both causal and bidirectional attention patterns.
    """
    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.causal = causal
        
        # Unified Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout - use different dropout rates based on config availability
        attn_dropout_rate = getattr(config, 'attn_pdrop', getattr(config, 'dropout', 0.1))
        resid_dropout_rate = getattr(config, 'resid_pdrop', getattr(config, 'dropout', 0.1))
        
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.resid_dropout = nn.Dropout(resid_dropout_rate)
        
        # Register causal mask buffer if needed
        if causal:
            # This will be filled in during forward pass based on sequence length
            self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate Q, K, V for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)  
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask if needed (for autoregressive models)
        if self.causal:
            if self.causal_mask is None or self.causal_mask.size(0) < T:
                # Create causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                self.register_buffer("causal_mask", causal_mask, persistent=False)
            att = att.masked_fill(~self.causal_mask[:T, :T], float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # Convert attention mask to shape (B, 1, 1, T) for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, n_head, T, T) x (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        
        # Concatenate heads: (B, n_head, T, head_dim) -> (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class BidirectionalSelfAttention(UnifiedMultiHeadAttention):
    """
    Bidirectional self-attention layer using the unified attention mechanism.
    This is a wrapper for backward compatibility with existing transformer code.
    """
    def __init__(self, config):
        super().__init__(config, causal=False)  # Bidirectional = non-causal

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # ModuleDict does not expose keys as attributes; use indexing
        self.mlpf = lambda x: m['dropout'](m['c_proj'](m['act'](m['c_fc'](x)))) # MLP forward

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class LSTMAttentionBlock(nn.Module):
    """
    A single LSTM layer with self-attention mechanism using the unified attention implementation.
    Each block consists of:
    1. LSTM layer
    2. Unified multi-head self-attention 
    3. Residual connection and layer normalization
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        
        # LSTM layer
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, 
                           batch_first=True, dropout=0)
        
        # Unified multi-head self-attention (same as used in transformer)
        self.attn = UnifiedMultiHeadAttention(config, causal=False)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        
    def forward(self, x, attention_mask=None):
        # Store residual for later
        residual = x
        
        # 1. LSTM processing with residual connection
        lstm_out, _ = self.lstm(x)
        x = self.ln1(lstm_out + residual)
        
        # 2. Multi-head self-attention with residual connection
        residual = x
        attn_out = self.attn(x, attention_mask)
        x = self.ln2(attn_out + residual)
        
        return x

class LSTMClassifier(nn.Module):
    """ LSTM-based text classifier with per-layer attention mechanism (similar to transformer architecture)"""
    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_embd) must be given in the config
        C.model_type = 'lstm'
        C.n_layer = None
        C.n_embd = None
        C.n_head = 4  # Number of attention heads
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters (compatible with GPT config names)
        C.dropout = 0.3
        C.embd_pdrop = 0.1  # Embedding dropout (same as GPT)
        C.resid_pdrop = 0.1  # Residual dropout (same as GPT) 
        C.attn_pdrop = 0.1   # Attention dropout (same as GPT)
        # classification specific
        C.num_classes = 2  # for binary sentiment classification
        return C
    
    def __init__(self, config):
        super().__init__()
        # basic sanity checks
        assert config.vocab_size is not None, "config.vocab_size must be set for LSTMClassifier"
        assert config.block_size is not None, "config.block_size must be set for LSTMClassifier"
        assert config.n_layer is not None, "config.n_layer must be set for LSTMClassifier"
        assert config.n_embd is not None, "config.n_embd must be set for LSTMClassifier"

        self.config = config
        
        # Embedding layer with dropout (matching GPT architecture)
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.embd_dropout = nn.Dropout(getattr(config, 'embd_pdrop', config.dropout))
        
        # Stack of LSTM-Attention blocks
        self.layers = nn.ModuleList([
            LSTMAttentionBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Dropout and classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_classes)
        
        # Language modeling head for pre-training (shares weights with embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between word embeddings and language modeling head (same as GPT)
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("LSTM with Attention: number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask, labels=None):
        # Embed tokens with dropout (matching GPT architecture)
        x = self.embedding(input_ids)  # (batch, seq_len, n_embd)
        x = self.embd_dropout(x)
        
        # Ensure attention_mask is float on the same device/dtype
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=x.dtype, device=x.device)
            # Apply attention mask to embeddings (zero out padded tokens)
            x = x * attention_mask.unsqueeze(-1)
        
        # Pass through each LSTM-Attention block
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Global average pooling with attention mask
        if attention_mask is not None:
            # Apply attention mask and compute mean over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            denom = attention_mask.sum(dim=1, keepdim=True).to(x.dtype) + 1e-8
            pooled = x_masked.sum(dim=1) / denom
        else:
            # Simple mean pooling if no attention mask
            pooled = x.mean(dim=1)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
    
    def get_hidden_states(self, input_ids, attention_mask=None):
        """Get hidden states for language modeling"""
        # Embed tokens with dropout
        x = self.embedding(input_ids)
        x = self.embd_dropout(x)
        
        # Ensure attention_mask is float on the same device/dtype
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=x.dtype, device=x.device)
            # Apply attention mask to embeddings (zero out padded tokens)
            x = x * attention_mask.unsqueeze(-1)
        
        # Pass through each LSTM-Attention block
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        return x
    
    def forward_lm(self, input_ids, attention_mask=None):
        """Forward pass for language modeling (pre-training)"""
        hidden_states = self.get_hidden_states(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits


class GPT(nn.Module):
    """ GPT Language Model modified for sentiment classification """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # classification specific
        C.num_classes = 2  # for binary sentiment classification
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Classification head for fine-tuning
        self.classifier = nn.Linear(config.n_embd, config.num_classes)
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        # Language modeling head for pre-training
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between word embeddings and language modeling head
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("Transformer: number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # ensure attention_mask is float on same device/dtype when used for pooling or passed to blocks
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=x.dtype, device=x.device)
        for block in self.transformer.h:
            x = block(x, attention_mask)
        x = self.transformer.ln_f(x)
        
        # Global average pooling with attention mask
        if attention_mask is not None:
            # Apply attention mask and compute mean over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            denom = attention_mask.sum(dim=1, keepdim=True).to(x.dtype) + 1e-8
            pooled = x_masked.sum(dim=1) / denom
        else:
            # Simple mean pooling if no attention mask
            pooled = x.mean(dim=1)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
    
    def get_hidden_states(self, input_ids):
        """Get hidden states for language modeling (used during pre-training)."""
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward through transformer
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        return x
    
    def forward_lm(self, input_ids, attention_mask=None):
        """Forward pass for language modeling (next token prediction)."""
        hidden_states = self.get_hidden_states(input_ids)
        return self.lm_head(hidden_states)
