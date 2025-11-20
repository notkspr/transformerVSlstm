# LSTM vs GPT Sentiment Classification

This repository provides a unified framework for comparing LSTM (with attention) and GPT (Transformer) models on sentiment classification tasks, with support for pre-training on Wikipedia and fine-tuning on IMDB. The code is designed for fair comparison, efficient training, and easy extensibility.

## Features

- Unified attention mechanism for both LSTM and GPT
- Weight sharing between embedding and language modeling head
- Pre-training on Wikipedia (language modeling)
- Fine-tuning on IMDB (sentiment classification)
- Automatic checkpointing and results logging
- Matplotlib/Seaborn visualizations
- Interactive sentiment analysis
- Flexible configuration via command-line arguments

## Usage: `train.py`

### Basic Training Modes

- **Pre-training only:**

  ```bash
  python train.py --mode pretrain --model lstm --pretrain_epochs 3 --wiki_subset_size 10000
  python train.py --mode pretrain --model gpt --pretrain_epochs 3 --wiki_subset_size 10000
  ```

- **Fine-tuning only:**

  ```bash
  python train.py --mode finetune --model lstm --finetune_epochs 5 --load_pretrained checkpoints/lstm_pretrained_final.pt
  python train.py --mode finetune --model gpt --finetune_epochs 5 --load_pretrained checkpoints/gpt_pretrained_final.pt
  ```

- **Full pipeline (pre-train + fine-tune):**
  ```bash
  python train.py --mode both --model both --pretrain_epochs 3 --finetune_epochs 5 --wiki_subset_size 10000
  ```

### Key Arguments

| Argument             | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `--mode`             | `pretrain`, `finetune`, or `both`                              |
| `--model`            | `lstm`, `gpt`, or `both`                                       |
| `--pretrain_epochs`  | Number of pre-training epochs                                  |
| `--finetune_epochs`  | Number of fine-tuning epochs                                   |
| `--epochs`           | Number of epochs for both phases (overrides specific settings) |
| `--lr`               | Learning rate (overrides default)                              |
| `--batch_size`       | Batch size (default: 32)                                       |
| `--wiki_subset_size` | Number of Wikipedia articles for pre-training                  |
| `--load_pretrained`  | Path to pre-trained checkpoint for fine-tuning                 |

### Advanced Usage Examples

```bash
# Train LSTM only with custom parameters
python train.py --mode both --model lstm --pretrain_epochs 5 --finetune_epochs 3 --lr 2e-4 --batch_size 16

# Train both models with different epoch settings
python train.py --mode pretrain --model both --pretrain_epochs 8 --wiki_subset_size 25000
python train.py --mode finetune --model both --finetune_epochs 2 --load_pretrained checkpoints/lstm_pretrained_final.pt

# Quick training for testing
python train.py --mode both --model both --pretrain_epochs 1 --finetune_epochs 1 --wiki_subset_size 1000
```

## Evaluation Framework

### Comprehensive Evaluation (`evaluation.py`)

Evaluates all available model checkpoints and generates comparison plots:

```bash
python evaluation.py
```

**Note:** This provides unified evaluation across all checkpoints, but doesn't separate pre-training and fine-tuning phases.

### Separated Phase Evaluation (`quick_separated_eval.py`)

For **phase-specific analysis**, use this script to separate pre-training and fine-tuning results:

```bash
python quick_separated_eval.py
```

#### What the separated evaluation provides:

**🔄 Pre-training Phase Analysis:**

- **Focus:** Language modeling performance on Wikipedia articles
- **Metrics:** Language modeling loss, Perplexity
- **Visualization:** Pre-training loss progression across epochs
- **Purpose:** Evaluate language understanding development

**🎯 Fine-tuning Phase Analysis:**

- **Focus:** Sentiment classification performance on IMDB reviews
- **Metrics:** Accuracy, F1 Score, Precision, Recall
- **Visualization:** Classification performance comparison
- **Purpose:** Evaluate task-specific performance

#### Generated outputs:

**Comprehensive Evaluation:**

- `evaluation/evaluation_results.json` - Complete numerical results
- `evaluation/*_comparison.png` - Metric comparison plots
- `evaluation/confusion_matrices.png` - Confusion matrices for best models

**Separated Phase Evaluation:**

- `evaluation/separated_evaluation_results.json` - Phase-separated results
- `evaluation/pretraining_metrics_comparison.png` - Pre-training language modeling
- `evaluation/finetuning_metrics_comparison.png` - Fine-tuning classification
- `evaluation/phase_comparison_summary.png` - Comprehensive phase comparison

#### Key Insights from Separated Evaluation:

1. **Pre-training Performance:** Shows how well models learn language patterns
2. **Fine-tuning Performance:** Shows task-specific classification ability
3. **Training Dynamics:** Reveals the impact of pre-training → fine-tuning pipeline
4. **Architecture Comparison:** Fair comparison of LSTM vs GPT across both phases

## Project Structure

````
├── train.py                    # Main training script
├── evaluation.py              # Comprehensive evaluation framework
├── quick_separated_eval.py    # Phase-separated evaluation
├── model.py                   # LSTM and GPT model definitions
├── bpe.py                     # Byte-pair encoding tokenizer
├── utils.py                   # Utility functions
├── main.py                    # Interactive sentiment analysis demo
├── requirements.txt           # Dependencies
├── checkpoints/               # Saved model checkpoints
├── evaluation/                # Evaluation results and plots
├── results/                   # Training metrics and visualizations
└── logs/                      # Training logs

### Example: Custom Training

```bash
python train.py --mode both --model lstm --pretrain_epochs 2 --finetune_epochs 4 --batch_size 16 --wiki_subset_size 5000
````

### Outputs

- **Checkpoints:** Saved in `checkpoints/` after each phase
- **Results:** Metrics and plots in `results/`
- **Logs:** Training logs in `logs/`

### Requirements

- Python 3.8+
- PyTorch
- Datasets
- Matplotlib, Seaborn, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Details

- **LSTMClassifier:** Multi-layer LSTM with per-layer attention, unified attention block, weight sharing for language modeling
- **GPT:** Transformer with unified attention, weight sharing for language modeling

## Evaluation: `evaluation.py`

### Comprehensive Model Evaluation

```bash
# Full evaluation with interactive testing
python evaluation.py

# Quick evaluation (plots only, no interaction)
python quick_eval.py
```

### Features

- **Multi-checkpoint evaluation**: Evaluates all available model checkpoints
- **Comparative visualizations**: Side-by-side LSTM vs GPT performance plots
- **Metrics tracking**: Accuracy, F1-score, precision, recall, loss across epochs
- **Confusion matrices**: Visual comparison of model predictions
- **Interactive sentiment analysis**: Real-time text analysis with both models
- **Export functionality**: All plots saved to `evaluation/` folder
- **JSON results**: Detailed metrics saved for further analysis

### Generated Outputs

- `evaluation/accuracy_comparison.png` - Accuracy across checkpoints
- `evaluation/f1_score_comparison.png` - F1 scores comparison
- `evaluation/precision_comparison.png` - Precision metrics
- `evaluation/recall_comparison.png` - Recall performance
- `evaluation/loss_comparison.png` - Loss curves
- `evaluation/confusion_matrices.png` - Confusion matrices for best models
- `evaluation/comprehensive_comparison.png` - All metrics in one view
- `evaluation/evaluation_results.json` - Detailed numerical results

### Notes

- For fair comparison, both models use nearly identical parameter budgets
- Pre-training uses next-token prediction on Wikipedia
- Fine-tuning uses IMDB sentiment classification
- Interactive evaluation and visualizations available via `main.py` and `evaluation.py`

---

For more details, see the code and comments in `train.py`, `model.py`, `main.py`, and `evaluation.py`.
