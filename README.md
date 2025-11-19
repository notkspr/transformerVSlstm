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
| Argument                | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `--mode`               | `pretrain`, `finetune`, or `both`                                |
| `--model`              | `lstm`, `gpt`, or `both`                                         |
| `--pretrain_epochs`    | Number of pre-training epochs                                    |
| `--finetune_epochs`    | Number of fine-tuning epochs                                     |
| `--epochs`             | Number of epochs for both phases (overrides specific settings)   |
| `--lr`                 | Learning rate (overrides default)                                |
| `--batch_size`         | Batch size (default: 32)                                         |
| `--wiki_subset_size`   | Number of Wikipedia articles for pre-training                    |
| `--load_pretrained`    | Path to pre-trained checkpoint for fine-tuning                   |

### Example: Custom Training
```bash
python train.py --mode both --model lstm --pretrain_epochs 2 --finetune_epochs 4 --batch_size 16 --wiki_subset_size 5000
```

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

### Notes
- For fair comparison, both models use nearly identical parameter budgets
- Pre-training uses next-token prediction on Wikipedia
- Fine-tuning uses IMDB sentiment classification
- Interactive evaluation and visualizations available via `main.py`

---

For more details, see the code and comments in `train.py`, `model.py`, and `main.py`.
