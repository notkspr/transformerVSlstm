"""
This file is made to benchmark both LSTMClassifier and GPT (defined on model.py) on the IMDB dataset.
Uses bpe.py for tokenization.

"""

# import the sentiment analysis dataset
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]


# tokenize the dataset
from bpe import BPETokenizer
tokenizer = BPETokenizer()

# Define constants first
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
MAX_LEN = 512
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
VOCAB_SIZE = tokenizer.vocab_size + 2  # +2 for padding and unknown tokens


# Add padding and truncation to tokenization
def tokenize_function(examples):
    tokens = tokenizer.encode(examples["text"])
    # Pad or truncate to MAX_LEN
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    attention_mask = [1] * len(tokens) + [0] * (MAX_LEN - len(tokens))
    tokens = tokens + [0] * (MAX_LEN - len(tokens))  # pad with 0
    return {"input_ids": tokens, "attention_mask": attention_mask}

train_data = train_data.map(tokenize_function, batched=False)
test_data = test_data.map(tokenize_function, batched=False)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# import the models
from model import LSTMClassifier, GPT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

# set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create model configs with unified parameters
lstm_config = LSTMClassifier.get_default_config()
lstm_config.vocab_size = VOCAB_SIZE
lstm_config.block_size = HIDDEN_DIM
lstm_config.n_embd = EMBEDDING_DIM
lstm_config.n_layer = NUM_LAYERS
lstm_config.n_head = NUM_HEADS
# Use same dropout parameter names as GPT for consistency
lstm_config.dropout = DROPOUT
lstm_config.embd_pdrop = DROPOUT
lstm_config.resid_pdrop = DROPOUT
lstm_config.attn_pdrop = DROPOUT
lstm_config.num_classes = 2

gpt_config = GPT.get_default_config()
gpt_config.model_type = None  # Clear model_type to use individual parameters
gpt_config.vocab_size = VOCAB_SIZE
gpt_config.block_size = MAX_LEN
gpt_config.n_embd = EMBEDDING_DIM
gpt_config.n_head = NUM_HEADS
gpt_config.n_layer = NUM_LAYERS
gpt_config.embd_pdrop = DROPOUT
gpt_config.resid_pdrop = DROPOUT
gpt_config.attn_pdrop = DROPOUT
gpt_config.num_classes = 2

lstm_model = LSTMClassifier(lstm_config).to(device)
gpt_model = GPT(gpt_config).to(device)

criterion = nn.CrossEntropyLoss()
lstm_optimizer = Adam(lstm_model.parameters(), lr=LEARNING_RATE)
gpt_optimizer = Adam(gpt_model.parameters(), lr=LEARNING_RATE)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

def train_model(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, return_predictions=False):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            if return_predictions:
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            
    accuracy = total_correct / total_samples

    if return_predictions:
        # Calculate additional metrics
        f1 = f1_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        return accuracy, f1, precision, recall, all_predictions, all_labels
    
    return accuracy

# Initialize lists to store metrics for plotting
lstm_train_losses = []
lstm_test_accuracies = []
lstm_f1_scores = []
lstm_precisions = []
lstm_recalls = []
gpt_train_losses = []
gpt_test_accuracies = []
gpt_f1_scores = []
gpt_precisions = []
gpt_recalls = []
epochs_list = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epochs_list.append(epoch + 1)
    
    print("Training LSTM Model...")
    start_time = time.time()
    lstm_train_loss = train_model(lstm_model, lstm_optimizer, train_loader)
    lstm_train_time = time.time() - start_time
    print(f"LSTM Training Loss: {lstm_train_loss:.4f}, Time: {lstm_train_time:.2f}s")
    lstm_train_losses.append(lstm_train_loss)
    
    print("Evaluating LSTM Model...")
    start_time = time.time()
    lstm_accuracy, lstm_f1, lstm_precision, lstm_recall, _, _ = evaluate_model(lstm_model, test_loader, return_predictions=True)
    lstm_eval_time = time.time() - start_time
    print(f"LSTM Test Accuracy: {lstm_accuracy:.4f}, F1: {lstm_f1:.4f}, Precision: {lstm_precision:.4f}, Recall: {lstm_recall:.4f}, Time: {lstm_eval_time:.2f}s")
    lstm_test_accuracies.append(lstm_accuracy)
    lstm_f1_scores.append(lstm_f1)
    lstm_precisions.append(lstm_precision)
    lstm_recalls.append(lstm_recall)
    
    print("Training GPT Model...")
    start_time = time.time()
    gpt_train_loss = train_model(gpt_model, gpt_optimizer, train_loader)
    gpt_train_time = time.time() - start_time
    print(f"GPT Training Loss: {gpt_train_loss:.4f}, Time: {gpt_train_time:.2f}s")
    gpt_train_losses.append(gpt_train_loss)
    
    print("Evaluating GPT Model...")
    start_time = time.time()
    gpt_accuracy, gpt_f1, gpt_precision, gpt_recall, _, _ = evaluate_model(gpt_model, test_loader, return_predictions=True)
    gpt_eval_time = time.time() - start_time
    print(f"GPT Test Accuracy: {gpt_accuracy:.4f}, F1: {gpt_f1:.4f}, Precision: {gpt_precision:.4f}, Recall: {gpt_recall:.4f}, Time: {gpt_eval_time:.2f}s")
    gpt_test_accuracies.append(gpt_accuracy)
    gpt_f1_scores.append(gpt_f1)
    gpt_precisions.append(gpt_precision)
    gpt_recalls.append(gpt_recall)
    print("-" * 50)

# Final Results
print("Final Results:")
print(f"LSTM Test Accuracy: {lstm_accuracy:.4f}")
print(f"GPT Test Accuracy: {gpt_accuracy:.4f}")

# Get final predictions for confusion matrices
print("\nGenerating final confusion matrices...")
_, _, _, _, lstm_final_preds, lstm_final_labels = evaluate_model(lstm_model, test_loader, return_predictions=True)
_, _, _, _, gpt_final_preds, gpt_final_labels = evaluate_model(gpt_model, test_loader, return_predictions=True)

# Plot the results using matplotlib
def plot_training_results():
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid for plots
    x = np.arange(len(epochs_list))
    width = 0.35
    
    # Plot 1: Training Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(x - width/2, lstm_train_losses, width, label='LSTM', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, gpt_train_losses, width, label='GPT', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xticks(x)
    ax1.set_xticklabels(epochs_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x - width/2, lstm_test_accuracies, width, label='LSTM', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, gpt_test_accuracies, width, label='GPT', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy per Epoch')
    ax2.set_xticks(x)
    ax2.set_xticklabels(epochs_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: F1 Score
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x - width/2, lstm_f1_scores, width, label='LSTM', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, gpt_f1_scores, width, label='GPT', alpha=0.8, color='lightcoral')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score per Epoch')
    ax3.set_xticks(x)
    ax3.set_xticklabels(epochs_list)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Precision
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(x - width/2, lstm_precisions, width, label='LSTM', alpha=0.8, color='skyblue')
    ax4.bar(x + width/2, gpt_precisions, width, label='GPT', alpha=0.8, color='lightcoral')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision per Epoch')
    ax4.set_xticks(x)
    ax4.set_xticklabels(epochs_list)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Plot 5: Recall
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(x - width/2, lstm_recalls, width, label='LSTM', alpha=0.8, color='skyblue')
    ax5.bar(x + width/2, gpt_recalls, width, label='GPT', alpha=0.8, color='lightcoral')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Recall')
    ax5.set_title('Recall per Epoch')
    ax5.set_xticks(x)
    ax5.set_xticklabels(epochs_list)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # Plot 6: Final Metrics Comparison
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
    lstm_final_metrics = [lstm_test_accuracies[-1], lstm_f1_scores[-1], lstm_precisions[-1], lstm_recalls[-1]]
    gpt_final_metrics = [gpt_test_accuracies[-1], gpt_f1_scores[-1], gpt_precisions[-1], gpt_recalls[-1]]
    
    x_metrics = np.arange(len(metrics))
    ax6.bar(x_metrics - width/2, lstm_final_metrics, width, label='LSTM', alpha=0.8, color='skyblue')
    ax6.bar(x_metrics + width/2, gpt_final_metrics, width, label='GPT', alpha=0.8, color='lightcoral')
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Score')
    ax6.set_title('Final Metrics Comparison')
    ax6.set_xticks(x_metrics)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_results()

# Plot confusion matrices
def plot_confusion_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LSTM Confusion Matrix
    cm_lstm = confusion_matrix(lstm_final_labels, lstm_final_preds)
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    ax1.set_title('LSTM Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # GPT Confusion Matrix
    cm_gpt = confusion_matrix(gpt_final_labels, gpt_final_preds)
    sns.heatmap(cm_gpt, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    ax2.set_title('GPT Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrices()

# Interactive text evaluation section
def predict_sentiment(text, model, model_name):
    """Predict sentiment for a given text using the specified model"""
    model.eval()
    
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    
    attention_mask = [1] * len(tokens) + [0] * (MAX_LEN - len(tokens))
    tokens = tokens + [0] * (MAX_LEN - len(tokens))  # pad with 0
    
    # Convert to tensors and add batch dimension
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

def interactive_evaluation():
    """Interactive section for users to input text and see predictions"""
    print("\n" + "="*60)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*60)
    print("Enter text to analyze sentiment (type 'quit' to exit)")
    print("-"*60)
    
    while True:
        user_input = input("\nEnter your text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending the program...")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        print(f"\nAnalyzing: '{user_input}'")
        print("-" * 40)
        
        # Get predictions from both models
        lstm_sentiment, lstm_confidence = predict_sentiment(user_input, lstm_model, "LSTM")
        gpt_sentiment, gpt_confidence = predict_sentiment(user_input, gpt_model, "GPT")
        
        print(f"LSTM Prediction:  {lstm_sentiment} (confidence: {lstm_confidence:.3f})")
        print(f"GPT Prediction:   {gpt_sentiment} (confidence: {gpt_confidence:.3f})")
        
        # Show agreement/disagreement
        if lstm_sentiment == gpt_sentiment:
            print(f"✓ Both models agree: {lstm_sentiment}")
        else:
            print(f"✗ Models disagree - LSTM: {lstm_sentiment}, GPT: {gpt_sentiment}")

# Run interactive evaluation
interactive_evaluation()