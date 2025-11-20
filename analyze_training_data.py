#!/usr/bin/env python3
"""
Analyze stored training/evaluation data from train.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import datetime

class TrainingDataAnalyzer:
    """Analyze and visualize stored training data"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.evaluation_dir = Path("evaluation")
        self.evaluation_dir.mkdir(exist_ok=True)
    
    def load_training_data(self, model_name: str) -> Dict:
        """Load all training data for a specific model"""
        data = {
            'pretrain': [],
            'finetune': [],
            'summary': None
        }
        
        # Load pre-training evaluation data
        pretrain_file = self.results_dir / f"{model_name}_pretrain_evaluation.json"
        if pretrain_file.exists():
            with open(pretrain_file, 'r') as f:
                data['pretrain'] = json.load(f)
        
        # Load fine-tuning evaluation data
        finetune_file = self.results_dir / f"{model_name}_finetune_evaluation.json"
        if finetune_file.exists():
            with open(finetune_file, 'r') as f:
                data['finetune'] = json.load(f)
        
        # Load training summary
        summary_file = self.results_dir / f"{model_name}_training_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data['summary'] = json.load(f)
        
        return data
    
    def plot_pretraining_progress(self, pretrain_data: List[Dict], model_name: str):
        """Plot pre-training progress"""
        if not pretrain_data:
            print(f"No pre-training data available for {model_name}")
            return
        
        epochs = [d['epoch'] for d in pretrain_data]
        losses = [d['metrics']['train_loss'] for d in pretrain_data]
        perplexities = [d['metrics']['perplexity'] for d in pretrain_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        ax1.plot(epochs, losses, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Language Modeling Loss')
        ax1.set_title(f'{model_name.upper()} Pre-training Loss Progress')
        ax1.grid(True, alpha=0.3)
        
        # Add value annotations
        for x, y in zip(epochs, losses):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        # Perplexity plot
        ax2.plot(epochs, perplexities, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title(f'{model_name.upper()} Pre-training Perplexity Progress')
        ax2.grid(True, alpha=0.3)
        
        # Add value annotations
        for x, y in zip(epochs, perplexities):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / f'{model_name}_pretraining_progress.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_finetuning_progress(self, finetune_data: List[Dict], model_name: str):
        """Plot fine-tuning progress"""
        if not finetune_data:
            print(f"No fine-tuning data available for {model_name}")
            return
        
        epochs = [d['epoch'] for d in finetune_data]
        train_losses = [d['metrics']['train_loss'] for d in finetune_data]
        val_losses = [d['metrics']['val_loss'] for d in finetune_data]
        train_accs = [d['metrics']['train_acc'] for d in finetune_data]
        val_accs = [d['metrics']['val_acc'] for d in finetune_data]
        val_f1s = [d['metrics']['val_f1'] for d in finetune_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name.upper()} Fine-tuning Loss Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=6)
        ax2.plot(epochs, val_accs, 's-', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name.upper()} Fine-tuning Accuracy Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score plot
        ax3.plot(epochs, val_f1s, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title(f'{model_name.upper()} Validation F1 Score Progress')
        ax3.grid(True, alpha=0.3)
        
        # Combined metrics
        ax4.plot(epochs, val_accs, 'o-', label='Validation Accuracy', linewidth=2, markersize=6)
        ax4.plot(epochs, val_f1s, 's-', label='Validation F1', linewidth=2, markersize=6)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title(f'{model_name.upper()} Validation Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / f'{model_name}_finetuning_progress.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_training_report(self, model_name: str):\n        \"\"\"Generate comprehensive training report\"\"\"\n        data = self.load_training_data(model_name)\n        \n        print(f\"\\n{'='*60}\")\n        print(f\"TRAINING REPORT: {model_name.upper()}\")\n        print(f\"{'='*60}\")\n        \n        # Pre-training summary\n        if data['pretrain']:\n            print(f\"\\n🔄 PRE-TRAINING PHASE:\")\n            print(f\"   Total epochs: {len(data['pretrain'])}\")\n            \n            best_epoch = min(data['pretrain'], key=lambda x: x['metrics']['train_loss'])\n            print(f\"   Best epoch: {best_epoch['epoch']}\")\n            print(f\"   Best loss: {best_epoch['metrics']['train_loss']:.4f}\")\n            print(f\"   Best perplexity: {best_epoch['metrics']['perplexity']:.2f}\")\n            \n            # Training details\n            print(f\"   Learning rate: {data['pretrain'][0]['learning_rate']}\")\n            print(f\"   Batch size: {data['pretrain'][0]['batch_size']}\")\n            print(f\"   Dataset size: {data['pretrain'][0]['dataset_size']}\")\n        \n        # Fine-tuning summary\n        if data['finetune']:\n            print(f\"\\n🎯 FINE-TUNING PHASE:\")\n            print(f\"   Total epochs: {len(data['finetune'])}\")\n            \n            best_epoch = max(data['finetune'], key=lambda x: x['metrics']['val_f1'])\n            print(f\"   Best epoch: {best_epoch['epoch']}\")\n            print(f\"   Best validation accuracy: {best_epoch['metrics']['val_acc']:.4f}\")\n            print(f\"   Best validation F1: {best_epoch['metrics']['val_f1']:.4f}\")\n            print(f\"   Best validation precision: {best_epoch['metrics']['val_precision']:.4f}\")\n            print(f\"   Best validation recall: {best_epoch['metrics']['val_recall']:.4f}\")\n            \n            # Training details\n            print(f\"   Learning rate: {data['finetune'][0]['learning_rate']}\")\n            print(f\"   Batch size: {data['finetune'][0]['batch_size']}\")\n            print(f\"   Train dataset size: {data['finetune'][0]['train_dataset_size']}\")\n            print(f\"   Validation dataset size: {data['finetune'][0]['val_dataset_size']}\")\n        \n        # Training timeline\n        if data['pretrain'] or data['finetune']:\n            print(f\"\\n📅 TRAINING TIMELINE:\")\n            all_events = []\n            \n            for epoch_data in data['pretrain']:\n                timestamp = epoch_data['timestamp']\n                dt = datetime.datetime.fromtimestamp(timestamp)\n                all_events.append((dt, f\"Pre-train Epoch {epoch_data['epoch']}\", \n                                 f\"Loss: {epoch_data['metrics']['train_loss']:.4f}\"))\n            \n            for epoch_data in data['finetune']:\n                timestamp = epoch_data['timestamp']\n                dt = datetime.datetime.fromtimestamp(timestamp)\n                all_events.append((dt, f\"Fine-tune Epoch {epoch_data['epoch']}\", \n                                 f\"Val F1: {epoch_data['metrics']['val_f1']:.4f}\"))\n            \n            all_events.sort(key=lambda x: x[0])\n            \n            for dt, phase, metric in all_events:\n                print(f\"   {dt.strftime('%Y-%m-%d %H:%M:%S')} - {phase}: {metric}\")\n        \n        # Generate plots\n        if data['pretrain']:\n            self.plot_pretraining_progress(data['pretrain'], model_name)\n            print(f\"\\nPre-training progress plot saved: evaluation/{model_name}_pretraining_progress.png\")\n        \n        if data['finetune']:\n            self.plot_finetuning_progress(data['finetune'], model_name)\n            print(f\"Fine-tuning progress plot saved: evaluation/{model_name}_finetuning_progress.png\")\n    \n    def compare_models(self, model_names: List[str]):\n        \"\"\"Compare multiple models\"\"\"\n        print(f\"\\n{'='*80}\")\n        print(f\"MODEL COMPARISON: {' vs '.join([m.upper() for m in model_names])}\")\n        print(f\"{'='*80}\")\n        \n        comparison_data = {}\n        for model_name in model_names:\n            comparison_data[model_name] = self.load_training_data(model_name)\n        \n        # Compare pre-training\n        print(f\"\\n🔄 PRE-TRAINING COMPARISON:\")\n        print(f\"{'Model':<15} {'Best Loss':<12} {'Best Perplexity':<15} {'Epochs':<8}\")\n        print(\"-\" * 55)\n        \n        for model_name, data in comparison_data.items():\n            if data['pretrain']:\n                best = min(data['pretrain'], key=lambda x: x['metrics']['train_loss'])\n                print(f\"{model_name.upper():<15} {best['metrics']['train_loss']:<12.4f} \"\n                      f\"{best['metrics']['perplexity']:<15.2f} {len(data['pretrain']):<8}\")\n        \n        # Compare fine-tuning\n        print(f\"\\n🎯 FINE-TUNING COMPARISON:\")\n        print(f\"{'Model':<15} {'Best Acc':<12} {'Best F1':<12} {'Best Prec':<12} {'Best Rec':<12} {'Epochs':<8}\")\n        print(\"-\" * 80)\n        \n        for model_name, data in comparison_data.items():\n            if data['finetune']:\n                best = max(data['finetune'], key=lambda x: x['metrics']['val_f1'])\n                print(f\"{model_name.upper():<15} {best['metrics']['val_acc']:<12.4f} \"\n                      f\"{best['metrics']['val_f1']:<12.4f} \"\n                      f\"{best['metrics']['val_precision']:<12.4f} \"\n                      f\"{best['metrics']['val_recall']:<12.4f} \"\n                      f\"{len(data['finetune']):<8}\")\n    \n    def list_available_models(self) -> List[str]:\n        \"\"\"List all available models with training data\"\"\"\n        model_names = set()\n        \n        for file_path in self.results_dir.glob(\"*_evaluation.json\"):\n            # Extract model name from filename (e.g., \"lstm_pretrain_evaluation.json\")\n            parts = file_path.stem.split('_')\n            if len(parts) >= 3:  # model_phase_evaluation\n                model_name = '_'.join(parts[:-2])  # Everything except \"phase_evaluation\"\n                model_names.add(model_name)\n        \n        return list(model_names)\n\ndef main():\n    \"\"\"Main analysis function\"\"\"\n    analyzer = TrainingDataAnalyzer()\n    \n    # List available models\n    available_models = analyzer.list_available_models()\n    \n    if not available_models:\n        print(\"No training data found! Please run train.py first.\")\n        return\n    \n    print(f\"Available models with training data: {available_models}\")\n    \n    # Generate individual reports\n    for model_name in available_models:\n        analyzer.generate_training_report(model_name)\n    \n    # Compare models if multiple available\n    if len(available_models) > 1:\n        analyzer.compare_models(available_models)\n    \n    print(f\"\\n📊 Analysis complete! Plots saved to: evaluation/\")\n\nif __name__ == \"__main__\":\n    main()