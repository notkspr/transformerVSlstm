#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing LSTM and GPT models.
Loads results from multiple epochs, generates comparison graphs, and provides interactive testing.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LSTMClassifier, GPT
from bpe import BPETokenizer
from train import Config, IMDBDataset, WikiTextDataset, load_imdb_data, load_wikipedia_data, create_model

class ModelEvaluator:
    """Comprehensive model evaluation and comparison framework"""
    
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BPETokenizer()
        self.evaluation_dir = Path("evaluation")
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Load evaluation datasets
        self._load_evaluation_data()
    
    def count_parameters(self, model) -> int:
        """Count total number of parameters in the model"""
        return sum(p.numel() for p in model.parameters())
    
    def get_model_display_name(self, model_name: str) -> str:
        """Get the display name for the model (LSTM -> LSTM with Attention, GPT -> Transformer)"""
        if 'lstm' in model_name.lower():
            return 'LSTM with Attention'
        elif 'gpt' in model_name.lower():
            return 'Transformer'
        else:
            return model_name.upper()
    
    def _load_evaluation_data(self):
        """Load evaluation datasets"""
        # Load IMDB test data for fine-tuning evaluation
        print("Loading IMDB data for fine-tuning evaluation...")
        _, self.test_imdb = load_imdb_data()
        self.test_dataset = IMDBDataset(self.test_imdb, self.tokenizer, self.config.MAX_LEN)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Load Wikipedia data for pre-training evaluation
        print("Loading Wikipedia data for pre-training evaluation...")
        wiki_texts = load_wikipedia_data(200)  # Small subset for evaluation
        self.wiki_dataset = WikiTextDataset(wiki_texts, self.tokenizer, self.config.MAX_LEN - 1)
        self.wiki_loader = DataLoader(self.wiki_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        print(f"Evaluation setup complete. Using device: {self.device}")
        print(f"IMDB test dataset size: {len(self.test_dataset)} samples")
        print(f"Wikipedia eval dataset size: {len(self.wiki_dataset)} samples")
        print(f"Evaluation results will be saved to: {self.evaluation_dir}")
    
    def find_available_checkpoints(self) -> Dict[str, List[str]]:
        """Find all available model checkpoints categorized by training phase"""
        checkpoints_dir = Path("checkpoints")
        if not checkpoints_dir.exists():
            print("No checkpoints directory found!")
            return {}
        
        checkpoints = {
            'lstm_pretrained': [],   # LSTM pre-training checkpoints
            'lstm_finetuned': [],    # LSTM fine-tuning checkpoints
            'gpt_pretrained': [],    # GPT pre-training checkpoints
            'gpt_finetuned': []      # GPT fine-tuning checkpoints
        }
        
        for checkpoint_file in checkpoints_dir.glob("*.pt"):
            name = checkpoint_file.stem.lower()
            if 'lstm' in name:
                if 'pretrain' in name:
                    checkpoints['lstm_pretrained'].append(str(checkpoint_file))
                elif 'finetun' in name or 'best' in name:
                    checkpoints['lstm_finetuned'].append(str(checkpoint_file))
            elif 'gpt' in name:
                if 'pretrain' in name:
                    checkpoints['gpt_pretrained'].append(str(checkpoint_file))
                elif 'finetun' in name or 'best' in name:
                    checkpoints['gpt_finetuned'].append(str(checkpoint_file))
        
        # Sort by epoch number if available
        def extract_epoch_number(filename):
            import re
            # Look for patterns like "epoch_1", "epoch_2", etc.
            match = re.search(r'epoch[_-]?(\d+)', filename.lower())
            if match:
                return int(match.group(1))
            # If no epoch number, put at end (for "best" or "final" checkpoints)
            return float('inf')
        
        for key in checkpoints:
            checkpoints[key].sort(key=extract_epoch_number)
        
        print("\nAvailable checkpoints:")
        for model_type, files in checkpoints.items():
            print(f"  {model_type}: {len(files)} checkpoints")
        
        return checkpoints
    
    def load_model_checkpoint(self, model_type: str, checkpoint_path: str, for_pretraining: bool = False) -> torch.nn.Module:
        """Load model from checkpoint"""
        model = create_model(model_type, self.tokenizer, self.config, for_pretraining=for_pretraining)
        model = model.to(self.device)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible layers (skip incompatible ones like lm_head vs classifier)
            model_dict = model.state_dict()
            compatible_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                elif k.startswith('lm_head') and k.replace('lm_head', 'classifier') in model_dict:
                    # Skip lm_head parameters when loading for classification
                    continue
                else:
                    print(f"Skipping incompatible parameter: {k}")
            
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
            
            print(f"Loaded {len(compatible_dict)} parameters from {checkpoint_path}")
            return model
            
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
    
    def evaluate_pretrained_model(self, model: torch.nn.Module, model_name: str) -> Dict:
        """Evaluate pre-trained model on Wikipedia text using language modeling loss"""
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        total_params = self.count_parameters(model)
        display_name = self.get_model_display_name(model_name)
        
        with torch.no_grad():
            for batch in self.wiki_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Forward pass for language modeling
                if hasattr(model, 'forward_lm'):
                    logits = model.forward_lm(input_ids, attention_mask)
                else:
                    # Fallback: get hidden states and project to vocabulary
                    hidden_states = model.get_hidden_states(input_ids, attention_mask)
                    logits = model.lm_head(hidden_states)
                
                # Calculate language modeling loss
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
        
        avg_loss = total_loss / len(self.wiki_loader)
        perplexity = np.exp(avg_loss)
        
        return {
            'model_name': model_name,
            'display_name': display_name,
            'language_modeling_loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'total_parameters': total_params
        }
    
    def evaluate_model(self, model: torch.nn.Module, model_name: str) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        total_params = self.count_parameters(model)
        display_name = self.get_model_display_name(model_name)
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, loss = model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(F.softmax(logits, dim=-1).cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        avg_loss = total_loss / len(self.test_loader)
        
        return {
            'model_name': model_name,
            'display_name': display_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'loss': avg_loss,
            'total_parameters': total_params,
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits
        }
    
    def evaluate_all_checkpoints(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Evaluate all available checkpoints separated by training phase"""
        checkpoints = self.find_available_checkpoints()
        results = {
            'pretrained': {'lstm': [], 'gpt': []},
            'finetuned': {'lstm': [], 'gpt': []}
        }
        
        # Evaluate pre-trained LSTM checkpoints
        for checkpoint_path in checkpoints['lstm_pretrained']:
            print(f"\nEvaluating LSTM pre-trained checkpoint: {checkpoint_path}")
            model = self.load_model_checkpoint('lstm', checkpoint_path, for_pretraining=True)
            if model is not None:
                stem = Path(checkpoint_path).stem
                epoch_info = self.extract_epoch_info(stem)
                result = self.evaluate_pretrained_model(model, f"LSTM {epoch_info}")
                result['checkpoint_path'] = checkpoint_path
                result['epoch_info'] = epoch_info
                results['pretrained']['lstm'].append(result)
        
        # Evaluate fine-tuned LSTM checkpoints
        for checkpoint_path in checkpoints['lstm_finetuned']:
            print(f"\nEvaluating LSTM fine-tuned checkpoint: {checkpoint_path}")
            model = self.load_model_checkpoint('lstm', checkpoint_path, for_pretraining=False)
            if model is not None:
                stem = Path(checkpoint_path).stem
                epoch_info = self.extract_epoch_info(stem)
                result = self.evaluate_model(model, f"LSTM {epoch_info}")
                result['checkpoint_path'] = checkpoint_path
                result['epoch_info'] = epoch_info
                results['finetuned']['lstm'].append(result)
        
        # Evaluate pre-trained GPT checkpoints
        for checkpoint_path in checkpoints['gpt_pretrained']:
            print(f"\nEvaluating GPT pre-trained checkpoint: {checkpoint_path}")
            model = self.load_model_checkpoint('gpt', checkpoint_path, for_pretraining=True)
            if model is not None:
                stem = Path(checkpoint_path).stem
                epoch_info = self.extract_epoch_info(stem)
                result = self.evaluate_pretrained_model(model, f"GPT {epoch_info}")
                result['checkpoint_path'] = checkpoint_path
                result['epoch_info'] = epoch_info
                results['pretrained']['gpt'].append(result)
        
        # Evaluate fine-tuned GPT checkpoints
        for checkpoint_path in checkpoints['gpt_finetuned']:
            print(f"\nEvaluating GPT fine-tuned checkpoint: {checkpoint_path}")
            model = self.load_model_checkpoint('gpt', checkpoint_path, for_pretraining=False)
            if model is not None:
                stem = Path(checkpoint_path).stem
                epoch_info = self.extract_epoch_info(stem)
                result = self.evaluate_model(model, f"GPT {epoch_info}")
                result['checkpoint_path'] = checkpoint_path
                result['epoch_info'] = epoch_info
                results['finetuned']['gpt'].append(result)
        
        return results
    
    def extract_epoch_info(self, checkpoint_name: str) -> str:
        """Extract epoch information from checkpoint name for better labeling"""
        import re
        
        # Look for epoch numbers
        epoch_match = re.search(r'epoch[_-]?(\d+)', checkpoint_name.lower())
        if epoch_match:
            epoch_num = epoch_match.group(1)
            if 'pretrained' in checkpoint_name.lower():
                return f"(Pretrain Epoch {epoch_num})"
            elif 'finetuned' in checkpoint_name.lower():
                return f"(Finetune Epoch {epoch_num})"
            else:
                return f"(Epoch {epoch_num})"
        
        # Handle special cases
        if 'best' in checkpoint_name.lower():
            if 'pretrained' in checkpoint_name.lower():
                return "(Pretrain Best)"
            elif 'finetuned' in checkpoint_name.lower():
                return "(Finetune Best)"
            else:
                return "(Best)"
        elif 'final' in checkpoint_name.lower():
            if 'pretrained' in checkpoint_name.lower():
                return "(Pretrain Final)"
            else:
                return "(Final)"
        
        return f"({checkpoint_name})"
    
    def plot_pretraining_comparison(self, results: Dict[str, List[Dict]], metric: str, title: str):
        """Plot comparison of pre-training metrics"""
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        lstm_values = [r[metric] for r in results['lstm']]
        gpt_values = [r[metric] for r in results['gpt']]
        
        # Get parameter counts for labels
        lstm_params = results['lstm'][0]['total_parameters'] if results['lstm'] else 0
        gpt_params = results['gpt'][0]['total_parameters'] if results['gpt'] else 0
        
        # Use sequential numbering for x-axis
        lstm_epochs = list(range(1, len(lstm_values) + 1))
        gpt_epochs = list(range(1, len(gpt_values) + 1))
        
        # Plot lines with parameter counts in labels
        plt.plot(lstm_epochs, lstm_values, 'o-', label=f'LSTM with Attention ({lstm_params:,} params)', 
                linewidth=2, markersize=8, color='blue')
        plt.plot(gpt_epochs, gpt_values, 's-', label=f'Transformer ({gpt_params:,} params)', 
                linewidth=2, markersize=8, color='red')
        
        plt.xlabel('Pre-training Epoch', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.title(f'Pre-training {title} Comparison: LSTM with Attention vs Transformer', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (x, y) in enumerate(zip(lstm_epochs, lstm_values)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        for i, (x, y) in enumerate(zip(gpt_epochs, gpt_values)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / f'pretraining_{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_finetuning_comparison(self, results: Dict[str, List[Dict]], metric: str, title: str):
        """Plot comparison of fine-tuning metrics"""
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        lstm_values = [r[metric] for r in results['lstm']]
        gpt_values = [r[metric] for r in results['gpt']]
        
        # Get parameter counts for labels
        lstm_params = results['lstm'][0]['total_parameters'] if results['lstm'] else 0
        gpt_params = results['gpt'][0]['total_parameters'] if results['gpt'] else 0
        
        # Use sequential numbering for x-axis
        lstm_epochs = list(range(1, len(lstm_values) + 1))
        gpt_epochs = list(range(1, len(gpt_values) + 1))
        
        # Plot lines with parameter counts in labels
        plt.plot(lstm_epochs, lstm_values, 'o-', label=f'LSTM with Attention ({lstm_params:,} params)', 
                linewidth=2, markersize=8, color='blue')
        plt.plot(gpt_epochs, gpt_values, 's-', label=f'Transformer ({gpt_params:,} params)', 
                linewidth=2, markersize=8, color='red')
        
        plt.xlabel('Fine-tuning Checkpoint', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.title(f'Fine-tuning {title} Comparison: LSTM with Attention vs Transformer', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (x, y) in enumerate(zip(lstm_epochs, lstm_values)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        for i, (x, y) in enumerate(zip(gpt_epochs, gpt_values)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / f'finetuning_{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    

    
    def plot_confusion_matrices(self, finetuned_results: Dict[str, List[Dict]]):
        """Plot confusion matrices for best fine-tuned models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Find best models (highest F1 score) from fine-tuned results only
        best_lstm = max(finetuned_results['lstm'], key=lambda x: x['f1_score']) if finetuned_results['lstm'] else None
        best_gpt = max(finetuned_results['gpt'], key=lambda x: x['f1_score']) if finetuned_results['gpt'] else None
        
        models_data = [
            (best_lstm, 'LSTM with Attention', 0),
            (best_gpt, 'Transformer', 1)
        ]
        
        for model_data, model_name, idx in models_data:
            if model_data is None:
                continue
                
            cm = confusion_matrix(model_data['labels'], model_data['predictions'])
            
            # Create heatmap using matplotlib
            im = axes[idx].imshow(cm, cmap='Blues', aspect='auto')
            param_count = model_data.get('total_parameters', 0)
            axes[idx].set_title(f'{model_name} Confusion Matrix\n(F1: {model_data["f1_score"]:.3f}, {param_count:,} params)')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Negative', 'Positive'])
            axes[idx].set_yticklabels(['Negative', 'Positive'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = axes[idx].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_comparison(self, pretrained_results: Dict[str, List[Dict]], finetuned_results: Dict[str, List[Dict]]):
        """Create comprehensive comparison plots for both phases"""
        # Create separate plots for pre-training and fine-tuning
        
        # Pre-training comprehensive plot
        if pretrained_results['lstm'] or pretrained_results['gpt']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Language modeling loss
            if pretrained_results['lstm'] and pretrained_results['gpt']:
                lstm_loss = [r['language_modeling_loss'] for r in pretrained_results['lstm']]
                gpt_loss = [r['language_modeling_loss'] for r in pretrained_results['gpt']]
                lstm_epochs = list(range(1, len(lstm_loss) + 1))
                gpt_epochs = list(range(1, len(gpt_loss) + 1))
                
                lstm_params = pretrained_results['lstm'][0].get('total_parameters', 0) if pretrained_results['lstm'] else 0
                gpt_params = pretrained_results['gpt'][0].get('total_parameters', 0) if pretrained_results['gpt'] else 0
                
                axes[0].plot(lstm_epochs, lstm_loss, 'o-', label=f'LSTM with Attention ({lstm_params:,} params)', linewidth=2, markersize=6)
                axes[0].plot(gpt_epochs, gpt_loss, 's-', label=f'Transformer ({gpt_params:,} params)', linewidth=2, markersize=6)
                axes[0].set_xlabel('Pre-training Epoch')
                axes[0].set_ylabel('Language Modeling Loss')
                axes[0].set_title('Pre-training Loss Comparison')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Perplexity
                lstm_ppl = [r['perplexity'] for r in pretrained_results['lstm']]
                gpt_ppl = [r['perplexity'] for r in pretrained_results['gpt']]
                
                axes[1].plot(lstm_epochs, lstm_ppl, 'o-', label=f'LSTM with Attention ({lstm_params:,} params)', linewidth=2, markersize=6)
                axes[1].plot(gpt_epochs, gpt_ppl, 's-', label=f'Transformer ({gpt_params:,} params)', linewidth=2, markersize=6)
                axes[1].set_xlabel('Pre-training Epoch')
                axes[1].set_ylabel('Perplexity')
                axes[1].set_title('Pre-training Perplexity Comparison')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.evaluation_dir / 'pretraining_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Fine-tuning comprehensive plot
        if finetuned_results['lstm'] or finetuned_results['gpt']:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'loss']
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Loss']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                if i < 5:  # We have 5 metrics
                    lstm_values = [r[metric] for r in finetuned_results['lstm']]
                    gpt_values = [r[metric] for r in finetuned_results['gpt']]
                    
                    # Use sequential checkpoint numbering
                    lstm_epochs = list(range(1, len(lstm_values) + 1))
                    gpt_epochs = list(range(1, len(gpt_values) + 1))
                    
                    lstm_params = finetuned_results['lstm'][0].get('total_parameters', 0) if finetuned_results['lstm'] else 0
                    gpt_params = finetuned_results['gpt'][0].get('total_parameters', 0) if finetuned_results['gpt'] else 0
                    
                    axes[i].plot(lstm_epochs, lstm_values, 'o-', label=f'LSTM with Attention ({lstm_params:,} params)', linewidth=2, markersize=6)
                    axes[i].plot(gpt_epochs, gpt_values, 's-', label=f'Transformer ({gpt_params:,} params)', linewidth=2, markersize=6)
                    
                    axes[i].set_xlabel('Fine-tuning Checkpoint')
                    axes[i].set_ylabel(name)
                    axes[i].set_title(f'Fine-tuning {name} Comparison')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Use the last subplot for a summary table
            axes[5].axis('off')
            
            # Create summary table
            if finetuned_results['lstm'] and finetuned_results['gpt']:
                best_lstm = max(finetuned_results['lstm'], key=lambda x: x['f1_score'])
                best_gpt = max(finetuned_results['gpt'], key=lambda x: x['f1_score'])
                
                lstm_params = best_lstm.get('total_parameters', 0)
                gpt_params = best_gpt.get('total_parameters', 0)
                
                # Use shorter column headers to prevent overflow
                table_data = [
                    ['Metric', f'LSTM w/ Attention\n({lstm_params:,} params)', f'Transformer\n({gpt_params:,} params)'],
                    ['Accuracy', f"{best_lstm['accuracy']:.3f}", f"{best_gpt['accuracy']:.3f}"],
                    ['F1 Score', f"{best_lstm['f1_score']:.3f}", f"{best_gpt['f1_score']:.3f}"],
                    ['Precision', f"{best_lstm['precision']:.3f}", f"{best_gpt['precision']:.3f}"],
                    ['Recall', f"{best_lstm['recall']:.3f}", f"{best_gpt['recall']:.3f}"],
                    ['Loss', f"{best_lstm['loss']:.3f}", f"{best_gpt['loss']:.3f}"]
                ]
                
                table = axes[5].table(cellText=table_data[1:], colLabels=table_data[0], 
                                    loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)  # Reduced font size
                table.scale(1.2, 1.8)  # Better scaling proportions
                
                # Adjust column widths to prevent overflow
                cellDict = table.get_celld()
                for i in range(len(table_data)):
                    for j in range(len(table_data[0])):
                        cellDict[(i, j)].set_width(0.3)  # Set uniform column width
                        cellDict[(i, j)].set_height(0.15)  # Set row height
                        if i == 0:  # Header row
                            cellDict[(i, j)].set_facecolor('#E6E6FA')
                            cellDict[(i, j)].set_text_props(weight='bold')
                
                axes[5].set_title('Best Fine-tuned Model Comparison', fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(self.evaluation_dir / 'finetuning_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    
    def interactive_sentiment_analysis(self, finetuned_results: Dict[str, List[Dict]]):
        """Interactive sentiment analysis using fine-tuned models"""
        if not finetuned_results['lstm'] or not finetuned_results['gpt']:
            print("Need both LSTM and GPT fine-tuned models for interactive analysis")
            return
        
        # Load best fine-tuned models
        best_lstm_result = max(finetuned_results['lstm'], key=lambda x: x['f1_score'])
        best_gpt_result = max(finetuned_results['gpt'], key=lambda x: x['f1_score'])
        
        lstm_model = self.load_model_checkpoint('lstm', best_lstm_result['checkpoint_path'], for_pretraining=False)
        gpt_model = self.load_model_checkpoint('gpt', best_gpt_result['checkpoint_path'], for_pretraining=False)
        
        if lstm_model is None or gpt_model is None:
            print("Failed to load models for interactive analysis")
            return
        
        lstm_model.eval()
        gpt_model.eval()
        
        print(f"\n{'='*60}")
        print("INTERACTIVE SENTIMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Using best models:")
        lstm_display = best_lstm_result.get('display_name', 'LSTM with Attention')
        gpt_display = best_gpt_result.get('display_name', 'Transformer')
        lstm_params = best_lstm_result.get('total_parameters', 0)
        gpt_params = best_gpt_result.get('total_parameters', 0)
        print(f"  {lstm_display}: {best_lstm_result['model_name']} (F1: {best_lstm_result['f1_score']:.3f}, {lstm_params:,} params)")
        print(f"  {gpt_display}: {best_gpt_result['model_name']} (F1: {best_gpt_result['f1_score']:.3f}, {gpt_params:,} params)")
        print("Enter text to analyze sentiment (type 'quit' to exit)")
        print("-" * 60)
        
        agreement_count = 0
        total_predictions = 0
        
        while True:
            user_input = input("\nEnter your text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            total_predictions += 1
            
            # Tokenize input
            tokens = self.tokenizer.encode(user_input)
            if len(tokens) > self.config.MAX_LEN:
                tokens = tokens[:self.config.MAX_LEN]
            
            # Pad tokens
            input_ids = tokens + [0] * (self.config.MAX_LEN - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (self.config.MAX_LEN - len(tokens))
            
            # Convert to tensors
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
            
            print(f"\nAnalyzing: '{user_input}'")
            print("-" * 40)
            
            # Get predictions
            with torch.no_grad():
                lstm_logits, _ = lstm_model(input_tensor, mask_tensor)
                gpt_logits, _ = gpt_model(input_tensor, mask_tensor)
                
                lstm_probs = F.softmax(lstm_logits, dim=-1)[0]
                gpt_probs = F.softmax(gpt_logits, dim=-1)[0]
                
                lstm_pred = torch.argmax(lstm_probs).item()
                gpt_pred = torch.argmax(gpt_probs).item()
                
                lstm_confidence = lstm_probs[lstm_pred].item()
                gpt_confidence = gpt_probs[gpt_pred].item()
                
                labels = ['Negative', 'Positive']
                
                print(f"LSTM with Attention Prediction:  {labels[lstm_pred]} (confidence: {lstm_confidence:.3f})")
                print(f"Transformer Prediction:          {labels[gpt_pred]} (confidence: {gpt_confidence:.3f})")
                
                if lstm_pred == gpt_pred:
                    agreement_count += 1
                    print(f"✓ Both models agree: {labels[lstm_pred]}")
                else:
                    print(f"✗ Models disagree - LSTM with Attention: {labels[lstm_pred]}, Transformer: {labels[gpt_pred]}")
        
        print(f"\nSession Summary:")
        print(f"Total predictions: {total_predictions}")
        if total_predictions > 0:
            agreement_rate = agreement_count / total_predictions
            print(f"Model agreement rate: {agreement_rate:.1%} ({agreement_count}/{total_predictions})")
    
    def save_evaluation_results(self, results: Dict[str, Dict[str, List[Dict]]]):
        """Save evaluation results to JSON file"""
        # Prepare data for JSON serialization (remove non-serializable items)
        json_results = {
            'pretrained': {'lstm': [], 'gpt': []},
            'finetuned': {'lstm': [], 'gpt': []}
        }
        
        # Save pre-training results
        for model_type in ['lstm', 'gpt']:
            for result in results['pretrained'][model_type]:
                json_result = {
                    'model_name': result['model_name'],
                    'display_name': result.get('display_name', result['model_name']),
                    'language_modeling_loss': float(result['language_modeling_loss']),
                    'perplexity': float(result['perplexity']),
                    'total_tokens': int(result['total_tokens']),
                    'total_parameters': int(result.get('total_parameters', 0)),
                    'checkpoint_path': result['checkpoint_path']
                }
                json_results['pretrained'][model_type].append(json_result)
        
        # Save fine-tuning results
        for model_type in ['lstm', 'gpt']:
            for result in results['finetuned'][model_type]:
                json_result = {
                    'model_name': result['model_name'],
                    'display_name': result.get('display_name', result['model_name']),
                    'accuracy': float(result['accuracy']),
                    'f1_score': float(result['f1_score']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'loss': float(result['loss']),
                    'total_parameters': int(result.get('total_parameters', 0)),
                    'checkpoint_path': result['checkpoint_path']
                }
                json_results['finetuned'][model_type].append(json_result)
        
        # Save to JSON
        with open(self.evaluation_dir / 'evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nEvaluation results saved to {self.evaluation_dir / 'evaluation_results.json'}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        # Evaluate all checkpoints
        results = self.evaluate_all_checkpoints()
        
        pretrained_results = results['pretrained']
        finetuned_results = results['finetuned']
        
        if (not pretrained_results['lstm'] and not pretrained_results['gpt'] and 
            not finetuned_results['lstm'] and not finetuned_results['gpt']):
            print("No model checkpoints found for evaluation!")
            return
        
        # Generate all comparison plots
        print("\nGenerating evaluation plots...")
        
        # Pre-training plots
        if pretrained_results['lstm'] and pretrained_results['gpt']:
            print("Generating pre-training comparison plots...")
            self.plot_pretraining_comparison(pretrained_results, 'language_modeling_loss', 'Language Modeling Loss')
            self.plot_pretraining_comparison(pretrained_results, 'perplexity', 'Perplexity')
        
        # Fine-tuning plots
        if finetuned_results['lstm'] and finetuned_results['gpt']:
            print("Generating fine-tuning comparison plots...")
            self.plot_finetuning_comparison(finetuned_results, 'accuracy', 'Accuracy')
            self.plot_finetuning_comparison(finetuned_results, 'f1_score', 'F1 Score')
            self.plot_finetuning_comparison(finetuned_results, 'precision', 'Precision')
            self.plot_finetuning_comparison(finetuned_results, 'recall', 'Recall')
            self.plot_finetuning_comparison(finetuned_results, 'loss', 'Classification Loss')
            
            self.plot_confusion_matrices(finetuned_results)
        
        # Comprehensive comparison
        self.plot_comprehensive_comparison(pretrained_results, finetuned_results)
        
        # Save results
        self.save_evaluation_results(results)
        
        # Print summary
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Pre-training summary
        print("\nPRE-TRAINING RESULTS:")
        print("-" * 40)
        for model_type in ['lstm', 'gpt']:
            if pretrained_results[model_type]:
                best_model = min(pretrained_results[model_type], key=lambda x: x['language_modeling_loss'])
                display_name = best_model.get('display_name', model_type.upper())
                param_count = best_model.get('total_parameters', 0)
                print(f"{display_name} Best Pre-trained Model ({param_count:,} params):")
                print(f"  Language Modeling Loss: {best_model['language_modeling_loss']:.4f}")
                print(f"  Perplexity: {best_model['perplexity']:.2f}")
                print()
        
        # Fine-tuning summary
        print("FINE-TUNING RESULTS:")
        print("-" * 40)
        for model_type in ['lstm', 'gpt']:
            if finetuned_results[model_type]:
                best_model = max(finetuned_results[model_type], key=lambda x: x['f1_score'])
                display_name = best_model.get('display_name', model_type.upper())
                param_count = best_model.get('total_parameters', 0)
                print(f"{display_name} Best Fine-tuned Model ({param_count:,} params):")
                print(f"  Accuracy: {best_model['accuracy']:.3f}")
                print(f"  F1 Score: {best_model['f1_score']:.3f}")
                print(f"  Precision: {best_model['precision']:.3f}")
                print(f"  Recall: {best_model['recall']:.3f}")
                print()
        
        print(f"All evaluation plots saved to: {self.evaluation_dir}")
        
        # Interactive testing using fine-tuned models
        if finetuned_results['lstm'] or finetuned_results['gpt']:
            print("\nStarting interactive sentiment analysis...")
            self.interactive_sentiment_analysis(finetuned_results)


def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()