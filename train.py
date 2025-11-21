#!/usr/bin/env python3
"""
Comprehensive training script for LSTM and GPT models with pre-training and fine-tuning.

Usage:
  python train.py --mode pretrain --model lstm --epochs 3
  python train.py --mode finetune --model gpt --epochs 5 --load_pretrained checkpoints/lstm_pretrained.pt
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

from model import LSTMClassifier, GPT
from bpe import BPETokenizer

# Global configuration
class Config:
    # Model hyperparameters
    BATCH_SIZE = 32
    PRETRAIN_EPOCHS = 8
    FINETUNE_EPOCHS = 3
    PRETRAIN_LR = 5e-4
    FINETUNE_LR = 1e-4
    MAX_LEN = 512
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 1
    NUM_HEADS = 2
    DROPOUT = 0.1
    
    # Data configuration
    WIKI_SUBSET_SIZE = 500  # Number of Wikipedia articles for pre-training
    
    # Paths
    CHECKPOINT_DIR = Path("checkpoints")
    LOGS_DIR = Path("logs")
    RESULTS_DIR = Path("results")

class WikiTextDataset(Dataset):
    """Dataset for Wikipedia pre-training with next-token prediction"""
    
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print(f"Processing {len(texts)} Wikipedia texts...")
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text)
            
            # Create sliding windows for language modeling
            for i in range(0, len(tokens) - max_len, max_len // 2):
                sequence = tokens[i:i + max_len]
                if len(sequence) == max_len:
                    self.data.append(sequence)
        
        print(f"Created {len(self.data)} training sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # For language modeling: input is sequence[:-1], target is sequence[1:]
        input_ids = sequence[:-1]
        targets = sequence[1:]
        
        # Create attention mask (all 1s for valid tokens)
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(targets, dtype=torch.long)
        }

class IMDBDataset(Dataset):
    """Dataset for IMDB sentiment classification fine-tuning"""
    
    def __init__(self, dataset, tokenizer: BPETokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print(f"Processing {len(dataset)} IMDB examples...")
        for example in tqdm(dataset, desc="Tokenizing IMDB"):
            tokens = tokenizer.encode(example['text'])
            
            # Truncate or pad to max_len
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            tokens = tokens + [0] * (max_len - len(tokens))
            
            self.data.append({
                'input_ids': tokens,
                'attention_mask': attention_mask,
                'label': example['label']
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
            'label': torch.tensor(self.data[idx]['label'], dtype=torch.long)
        }

class Trainer:
    """Unified trainer for both pre-training and fine-tuning"""
    
    def __init__(self, model, tokenizer, device, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Create directories
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.config.LOGS_DIR.mkdir(exist_ok=True)
        self.config.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Initialize evaluation tracking
        self.evaluation_history = {
            'pretrain': [],  # Pre-training evaluation data
            'finetune': []   # Fine-tuning evaluation data
        }
        self.model_name = type(self.model).__name__.lower()
        
        # Get model parameter count and display name
        self.total_parameters = self.count_parameters()
        self.display_name = self.get_model_display_name()
        
        # Initialize timing tracking
        self.training_start_time = None
        self.epoch_times = []
        
        print(f"Initialized {self.display_name} with {self.total_parameters:,} parameters")
    
    def count_parameters(self) -> int:
        """Count total number of parameters in the model"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_display_name(self) -> str:
        """Get the display name for the model (LSTM -> LSTM with Attention, GPT -> Transformer)"""
        if 'lstm' in self.model_name:
            return 'LSTM with Attention'
        elif 'gpt' in self.model_name:
            return 'Transformer'
        else:
            return self.model_name.upper()
        
    def pretrain(self, wiki_texts: List[str], epochs: int, lr: float) -> Dict[str, List[float]]:
        """Pre-train model on Wikipedia text using language modeling objective"""
        print(f"\n{'='*60}")
        print(f"PRE-TRAINING {self.display_name} MODEL")
        print(f"{'='*60}")
        
        # Create dataset and dataloader
        dataset = WikiTextDataset(wiki_texts, self.tokenizer, self.config.MAX_LEN - 1)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Setup optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Training metrics
        metrics = {'train_loss': [], 'perplexity': [], 'epoch_times': []}
        
        # Start timing
        self.training_start_time = time.time()
        
        self.model.train()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_tokens = 0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['labels'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass - get logits for next token prediction
                    if hasattr(self.model, 'forward_lm'):  # Language modeling forward
                        logits = self.model.forward_lm(input_ids, attention_mask)
                    else:
                        # Fallback: use regular forward and extract hidden states
                        hidden_states = self.model.get_hidden_states(input_ids, attention_mask)
                        # Project to vocabulary for next token prediction
                        logits = self.model.lm_head(hidden_states)
                    
                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    epoch_tokens += attention_mask.sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'ppl': f"{torch.exp(loss):.2f}"
                    })
            
            # Calculate epoch metrics and timing
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - self.training_start_time
            
            avg_loss = epoch_loss / len(dataloader)
            perplexity = np.exp(avg_loss)
            
            metrics['train_loss'].append(avg_loss)
            metrics['perplexity'].append(perplexity)
            metrics['epoch_times'].append(elapsed_time)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}, Time={epoch_time:.1f}s (Total: {elapsed_time:.1f}s)")
            
            # Save evaluation data
            epoch_metrics = {
                'train_loss': avg_loss,
                'perplexity': perplexity,
                'total_tokens': epoch_tokens,
                'epoch_time': epoch_time,
                'elapsed_time': elapsed_time
            }
            additional_data = {
                'learning_rate': lr,
                'batch_size': self.config.BATCH_SIZE,
                'dataset_size': len(dataloader.dataset),
                'total_parameters': self.total_parameters,
                'model_display_name': self.display_name
            }
            self.save_evaluation_data('pretrain', epoch+1, epoch_metrics, additional_data)
            
            # Save checkpoint
            self.save_checkpoint(f"{type(self.model).__name__.lower()}_pretrained_epoch_{epoch+1}.pt", epoch+1)
        
        # Save final pre-training summary
        summary = self.create_evaluation_summary()
        summary_file = self.config.RESULTS_DIR / f"{self.model_name}_pretrain_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Pre-training summary saved: {summary_file}")
        
        return metrics
    
    def finetune(self, train_loader, val_loader, epochs: int, lr: float) -> Dict[str, List[float]]:
        """Fine-tune model on IMDB sentiment classification"""
        print(f"\n{'='*60}")
        print(f"FINE-TUNING {self.display_name} MODEL")
        print(f"{'='*60}")
        
        optimizer = Adam(self.model.parameters(), lr=lr)
        
        metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'epoch_times': []
        }
        
        # Start timing
        self.training_start_time = time.time()
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}") as pbar:
                for batch in pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    logits, loss = self.model(input_ids, attention_mask, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{train_correct/train_total:.4f}"
                    })
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - self.training_start_time
            
            # Store metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc_avg = train_correct / train_total
            
            metrics['train_loss'].append(train_loss_avg)
            metrics['train_acc'].append(train_acc_avg)
            metrics['val_loss'].append(val_metrics['loss'])
            metrics['val_acc'].append(val_metrics['accuracy'])
            metrics['val_f1'].append(val_metrics['f1'])
            metrics['epoch_times'].append(elapsed_time)
            
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {epoch_time:.1f}s (Total: {elapsed_time:.1f}s)")
            
            # Save evaluation data
            epoch_metrics = {
                'train_loss': train_loss_avg,
                'train_acc': train_acc_avg,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'epoch_time': epoch_time,
                'elapsed_time': elapsed_time
            }
            additional_data = {
                'learning_rate': lr,
                'batch_size': self.config.BATCH_SIZE,
                'train_dataset_size': len(train_loader.dataset),
                'val_dataset_size': len(val_loader.dataset),
                'is_best_epoch': val_metrics['accuracy'] > best_val_acc,
                'total_parameters': self.total_parameters,
                'model_display_name': self.display_name
            }
            self.save_evaluation_data('finetune', epoch+1, epoch_metrics, additional_data)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(f"{type(self.model).__name__.lower()}_finetuned_best.pt", epoch+1, is_best=True)
            
            # Save checkpoint for this epoch
                self.save_checkpoint(f"{type(self.model).__name__.lower()}_finetuned_best.pt", epoch+1, is_best=True)
            
            # Save checkpoint for this epoch
            self.save_checkpoint(f"{type(self.model).__name__.lower()}_finetuned_epoch_{epoch+1}.pt", epoch+1)
        
        # Save final training summary
        summary = self.create_evaluation_summary()
        summary_file = self.config.RESULTS_DIR / f"{self.model_name}_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved: {summary_file}")
        
        return metrics
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, loss = self.model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def save_checkpoint(self, filename: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config if hasattr(self.model, 'config') else None,
            'model_type': type(self.model).__name__
        }
        
        filepath = self.config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / f"{type(self.model).__name__.lower()}_best.pt"
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {filepath}")
    
    def save_evaluation_data(self, phase: str, epoch: int, metrics: Dict, additional_data: Dict = None):
        """Save evaluation data for current epoch"""
        # Convert metrics to ensure JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                serializable_metrics[key] = float(value) if isinstance(value, (int, float)) else value
            else:
                serializable_metrics[key] = str(value)
        
        eval_data = {
            'epoch': epoch,
            'phase': phase,
            'model_name': self.model_name,
            'timestamp': time.time(),
            'metrics': serializable_metrics
        }
        
        # Add any additional data (e.g., learning rate, batch size, etc.)
        if additional_data:
            serializable_additional = {}
            for key, value in additional_data.items():
                if isinstance(value, (int, float, str, bool)):
                    serializable_additional[key] = value
                else:
                    serializable_additional[key] = str(value)
            eval_data.update(serializable_additional)
        
        # Store in history
        self.evaluation_history[phase].append(eval_data)
        
        # Save to JSON file immediately (incremental save)
        eval_file = self.config.RESULTS_DIR / f"{self.model_name}_{phase}_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_history[phase], f, indent=2)
        
        print(f"Evaluation data saved: {eval_file}")
    
    def create_evaluation_summary(self) -> Dict:
        """Create comprehensive evaluation summary"""
        summary = {
            'model_name': self.model_name,
            'total_pretrain_epochs': len(self.evaluation_history['pretrain']),
            'total_finetune_epochs': len(self.evaluation_history['finetune']),
            'pretrain_history': self.evaluation_history['pretrain'],
            'finetune_history': self.evaluation_history['finetune']
        }
        
        # Calculate best performances
        if self.evaluation_history['pretrain']:
            best_pretrain = min(self.evaluation_history['pretrain'], 
                              key=lambda x: x['metrics']['train_loss'])
            summary['best_pretrain'] = {
                'epoch': best_pretrain['epoch'],
                'loss': best_pretrain['metrics']['train_loss'],
                'perplexity': best_pretrain['metrics']['perplexity']
            }
        
        if self.evaluation_history['finetune']:
            best_finetune = max(self.evaluation_history['finetune'], 
                              key=lambda x: x['metrics']['val_f1'])
            summary['best_finetune'] = {
                'epoch': best_finetune['epoch'],
                'accuracy': best_finetune['metrics']['val_acc'],
                'f1_score': best_finetune['metrics']['val_f1'],
                'precision': best_finetune['metrics'].get('val_precision', 0),
                'recall': best_finetune['metrics'].get('val_recall', 0)
            }
        
        return summary
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        if not strict:
            # Load only compatible layers (skip classifier/lm_head mismatches)
            model_dict = self.model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            
            # Filter out incompatible keys
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    print(f"Skipping incompatible layer: {k} (shapes: model={model_dict.get(k, 'missing').shape if k in model_dict else 'missing'}, checkpoint={v.shape})")
            
            # Update model dict and load
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict)
            print(f"Loaded {len(compatible_dict)} compatible layers from checkpoint")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}: {filepath}")
        return epoch

def create_model(model_type: str, tokenizer: BPETokenizer, config: Config, for_pretraining: bool = False):
    """Create and configure model for either pre-training or fine-tuning"""
    vocab_size = tokenizer.vocab_size + 2
    
    if model_type.lower() == 'lstm':
        model_config = LSTMClassifier.get_default_config()
        model_config.vocab_size = vocab_size
        model_config.block_size = config.HIDDEN_DIM
        model_config.n_embd = config.EMBEDDING_DIM
        model_config.n_layer = config.NUM_LAYERS
        model_config.n_head = config.NUM_HEADS
        model_config.dropout = config.DROPOUT
        model_config.embd_pdrop = config.DROPOUT
        model_config.resid_pdrop = config.DROPOUT
        model_config.attn_pdrop = config.DROPOUT
        model_config.num_classes = 2  # Always 2 for classification head
        
        model = LSTMClassifier(model_config)
        
    elif model_type.lower() == 'gpt':
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.vocab_size = vocab_size
        model_config.block_size = config.MAX_LEN
        model_config.n_embd = config.EMBEDDING_DIM
        model_config.n_head = config.NUM_HEADS
        model_config.n_layer = config.NUM_LAYERS
        model_config.embd_pdrop = config.DROPOUT
        model_config.resid_pdrop = config.DROPOUT
        model_config.attn_pdrop = config.DROPOUT
        model_config.num_classes = 2  # Always 2 for classification head
        
        model = GPT(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def load_wikipedia_data(subset_size: int = 50000) -> List[str]:
    """Load Wikipedia dataset for pre-training"""
    print(f"Loading Wikipedia dataset (subset of {subset_size} articles)...")
    
    try:
        # Load Wikipedia dataset
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{subset_size}]")
        texts = [article['text'] for article in wiki_dataset if len(article['text']) > 100]
        print(f"Loaded {len(texts)} Wikipedia articles")
        return texts
    except Exception as e:
        print(f"Error loading Wikipedia dataset: {e}")
        print("Using fallback smaller dataset...")
        # Fallback to a smaller dataset
        try:
            wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            texts = [article['text'] for article in wiki_dataset if len(article['text']) > 100]
            return texts[:subset_size]
        except:
            # Ultimate fallback - create dummy data
            print("Using dummy data for testing...")
            return [f"This is a sample Wikipedia article number {i} with some content." * 10 for i in range(1000)]

def load_imdb_data():
    """Load IMDB dataset for fine-tuning"""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    return dataset["train"], dataset["test"]

def plot_metrics(metrics: Dict[str, List[float]], model_name: str, phase: str, total_params: int = None):
    """Plot training metrics including time-based plots"""
    # Convert model name if needed
    display_name = model_name
    if 'lstm' in model_name.lower():
        display_name = 'LSTM with Attention'
    elif 'gpt' in model_name.lower():
        display_name = 'Transformer'
    
    if phase == "pretrain":
        # Create 2x2 subplot for pretrain: Loss vs Epochs, Perplexity vs Epochs, Loss vs Time, Perplexity vs Time
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(metrics['train_loss']) + 1)
        times = metrics.get('epoch_times', epochs)  # Use times if available, else epochs
        
        # Loss vs Epochs (original)
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        title_suffix = f' ({total_params:,} params)' if total_params else ''
        ax1.set_title(f'{display_name} Pre-training Loss vs Epochs{title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perplexity vs Epochs (original)
        ax2.plot(epochs, metrics['perplexity'], 'r-', label='Perplexity', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title(f'{display_name} Pre-training Perplexity vs Epochs{title_suffix}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss vs Time (new)
        ax3.plot(times, metrics['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'{display_name} Pre-training Loss vs Time{title_suffix}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Perplexity vs Time (new)
        ax4.plot(times, metrics['perplexity'], 'r-', label='Perplexity', linewidth=2, marker='s')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Perplexity')
        ax4.set_title(f'{display_name} Pre-training Perplexity vs Time{title_suffix}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    else:  # finetune
        # Create 2x3 subplot for finetune: Loss vs Epochs, Accuracy vs Epochs, F1 vs Epochs, Loss vs Time, Accuracy vs Time, F1 vs Time
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = range(1, len(metrics['train_loss']) + 1)
        times = metrics.get('epoch_times', epochs)  # Use times if available, else epochs
        title_suffix = f' ({total_params:,} params)' if total_params else ''
        
        # Loss vs Epochs (original)
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{display_name} Fine-tuning Loss vs Epochs{title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs Epochs (original)
        ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
        ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{display_name} Fine-tuning Accuracy vs Epochs{title_suffix}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score vs Epochs (original)
        ax3.plot(epochs, metrics['val_f1'], 'g-', label='Validation F1', linewidth=2, marker='^')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title(f'{display_name} Fine-tuning F1 vs Epochs{title_suffix}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Loss vs Time (new)
        ax4.plot(times, metrics['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
        ax4.plot(times, metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Loss')
        ax4.set_title(f'{display_name} Fine-tuning Loss vs Time{title_suffix}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Accuracy vs Time (new)
        ax5.plot(times, metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
        ax5.plot(times, metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Accuracy')
        ax5.set_title(f'{display_name} Fine-tuning Accuracy vs Time{title_suffix}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # F1 Score vs Time (new)
        ax6.plot(times, metrics['val_f1'], 'g-', label='Validation F1', linewidth=2, marker='^')
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('F1 Score')
        ax6.set_title(f'{display_name} Fine-tuning F1 vs Time{title_suffix}')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{model_name}_{phase}_metrics.png"
    plt.savefig(Config.RESULTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train LSTM and GPT models with pre-training and fine-tuning')
    parser.add_argument('--mode', choices=['pretrain', 'finetune', 'both'], default='both',
                        help='Training mode: pretrain, finetune, or both')
    parser.add_argument('--model', choices=['lstm', 'gpt', 'both'], default='both',
                        help='Model to train: lstm, gpt, or both')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides defaults for both phases)')
    parser.add_argument('--pretrain_epochs', type=int, help='Number of pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides defaults)')
    parser.add_argument('--load_pretrained', type=str, help='Path to pre-trained model checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--wiki_subset_size', type=int, help='Wikipedia subset size for pre-training')
    
    args = parser.parse_args()
    
    # Initialize
    config = Config()
    config.BATCH_SIZE = args.batch_size if args.batch_size else config.BATCH_SIZE
    config.WIKI_SUBSET_SIZE = args.wiki_subset_size if args.wiki_subset_size else config.WIKI_SUBSET_SIZE
    
    # Handle epoch settings
    if args.epochs:
        # If general epochs specified, use for both phases
        pretrain_epochs = args.epochs
        finetune_epochs = args.epochs
    else:
        # Use specific epoch settings or defaults
        pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs else config.PRETRAIN_EPOCHS
        finetune_epochs = args.finetune_epochs if args.finetune_epochs else config.FINETUNE_EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BPETokenizer()
    
    # Determine models to train
    models_to_train = ['lstm', 'gpt'] if args.model == 'both' else [args.model]
    
    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"PROCESSING {model_name.upper()} MODEL")
        print(f"{'='*80}")
        
        # Pre-training phase
        if args.mode in ['pretrain', 'both']:
            # Create model for pre-training
            model = create_model(model_name, tokenizer, config, for_pretraining=True)
            model = model.to(device)
            
            trainer = Trainer(model, tokenizer, device, config)
            
            # Load Wikipedia data
            wiki_texts = load_wikipedia_data(config.WIKI_SUBSET_SIZE)
            
            # Pre-train
            lr = args.lr if args.lr else config.PRETRAIN_LR
            
            pretrain_metrics = trainer.pretrain(wiki_texts, pretrain_epochs, lr)
            
            # Plot and save results
            plot_metrics(pretrain_metrics, model_name.upper(), 'pretrain', trainer.total_parameters)
            
            # Save final checkpoint
            trainer.save_checkpoint(f"{model_name}_pretrained_final.pt", pretrain_epochs)
        
        # Fine-tuning phase
        if args.mode in ['finetune', 'both']:
            # Create model for fine-tuning
            model = create_model(model_name, tokenizer, config, for_pretraining=False)
            model = model.to(device)
            
            trainer = Trainer(model, tokenizer, device, config)
            
            # Load pre-trained weights if available
            if args.load_pretrained:
                trainer.load_checkpoint(args.load_pretrained)
            elif args.mode == 'both':
                # Load from just completed pre-training (skip incompatible layers)
                pretrained_path = config.CHECKPOINT_DIR / f"{model_name}_pretrained_final.pt"
                if pretrained_path.exists():
                    trainer.load_checkpoint(str(pretrained_path), strict=False)
            
            # Load IMDB data
            train_imdb, test_imdb = load_imdb_data()
            
            train_dataset = IMDBDataset(train_imdb, tokenizer, config.MAX_LEN)
            test_dataset = IMDBDataset(test_imdb, tokenizer, config.MAX_LEN)
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            # Fine-tuning
            lr = args.lr if args.lr else config.FINETUNE_LR
            
            finetune_metrics = trainer.finetune(train_loader, test_loader, finetune_epochs, lr)
            
            # Final evaluation
            final_results = trainer.evaluate(test_loader)
            
            print(f"\nFinal {model_name.upper()} Results:")
            print(f"  Accuracy: {final_results['accuracy']:.4f}")
            print(f"  F1 Score: {final_results['f1']:.4f}")
            print(f"  Precision: {final_results['precision']:.4f}")
            print(f"  Recall: {final_results['recall']:.4f}")
            
            # Plot metrics
            plot_metrics(finetune_metrics, model_name.upper(), 'finetune', trainer.total_parameters)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(final_results['labels'], final_results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            display_name = 'LSTM with Attention' if 'lstm' in model_name.lower() else 'Transformer' if 'gpt' in model_name.lower() else model_name.upper()
            plt.title(f'{display_name} Confusion Matrix ({trainer.total_parameters:,} params)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(config.RESULTS_DIR / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save results (simplified to avoid circular references)
            results_file = config.RESULTS_DIR / f"{model_name}_results.json"
            with open(results_file, 'w') as f:
                # Convert complex objects to simple dictionaries
                simple_results = {}
                for key, value in final_results.items():
                    if isinstance(value, (float, int, str)):
                        simple_results[key] = value
                    else:
                        simple_results[key] = str(value)
                
                json.dump({
                    'final_metrics': simple_results,
                    'model_name': model_name
                }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Checkpoints saved in: {config.CHECKPOINT_DIR}")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"Logs saved in: {config.LOGS_DIR}")

if __name__ == "__main__":
    main()