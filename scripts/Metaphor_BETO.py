# Split Data into Training, Validation, and Test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load the preprocessed data
df = pd.read_csv("combined_metaphors_nonmetaphors.csv")

# Check class distribution before splitting
print("Original dataset:")
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Class balance: {df['label'].value_counts(normalize=True)}")

# Split the data into training, validation, and testing sets (80% train, 10% validation, 10% test)
# Using stratify to maintain class balance across splits
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save the splits (optional)
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# Check the splits and class distribution
print(f"\nData splits:")
print(f"Training data: {len(train_df)} samples")
print(f"  - Metaphors: {sum(train_df['label'] == 1)}")
print(f"  - Non-metaphors: {sum(train_df['label'] == 0)}")

print(f"Validation data: {len(val_df)} samples")
print(f"  - Metaphors: {sum(val_df['label'] == 1)}")
print(f"  - Non-metaphors: {sum(val_df['label'] == 0)}")

print(f"Testing data: {len(test_df)} samples")
print(f"  - Metaphors: {sum(test_df['label'] == 1)}")
print(f"  - Non-metaphors: {sum(test_df['label'] == 0)}")

# Prepare Data for Hugging Face Pre-Trained Model
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

# Load the pre-trained Spanish BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Create a custom dataset class
class MetaphorDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the text and label for a particular row
        text = str(self.data.iloc[idx]['text'])  # Ensure text is string
        label = int(self.data.iloc[idx]['label'])  # Ensure label is int
        
        # Tokenize the text and encode it in the BERT format
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Adds [CLS] and [SEP] tokens
            max_length=self.max_length,  # Pad or truncate to max_length
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # We also need the attention mask
            return_tensors='pt',  # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets for train, validation, and test
train_dataset = MetaphorDataset(train_df, tokenizer)
val_dataset = MetaphorDataset(val_df, tokenizer)
test_dataset = MetaphorDataset(test_df, tokenizer)

# Check the shape of the first example
print(f"\nExample tokenized input: {train_dataset[0]}")
print(f"Input IDs shape: {train_dataset[0]['input_ids'].shape}")
print(f"Attention mask shape: {train_dataset[0]['attention_mask'].shape}")
print(f"Label: {train_dataset[0]['labels'].item()}")

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1-score during evaluation
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Fine-tune the pre-trained BERT model for metaphor identification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased",
    num_labels=2,  # Binary classification (metaphor vs non-metaphor)
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=500,  # Evaluate every 500 steps (or adjust as needed)
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,  # Explicit learning rate
    warmup_steps=100,  # Learning rate warmup
    save_total_limit=2,  # Only keep 2 best checkpoints
    report_to=None,  # Don't report to wandb/tensorboard# Add other arguments you need
)

# Initialize Trainer with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # This is used during training for evaluation
    compute_metrics=compute_metrics,  # This computes our custom metrics
)

# Fine-tune the model
print("\nStarting training...")
trainer.train()

# Evaluate the model on the validation set (final evaluation)
print("\nFinal evaluation on validation set:")
val_results = trainer.evaluate(val_dataset)
print(val_results)

# NOW EVALUATE ON TEST SET (previously unseen data)
print("\nEvaluating on test set (unseen data):")
test_results = trainer.evaluate(test_dataset)
print("Test Results:")
print(test_results)

# Get detailed predictions on test set
print("\nGetting detailed predictions on test set...")
test_predictions = trainer.predict(test_dataset)
y_pred = np.argmax(test_predictions.predictions, axis=1)
y_true = test_predictions.label_ids

# Calculate detailed metrics
test_accuracy = accuracy_score(y_true, y_pred)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"\nDetailed Test Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Classification report with per-class metrics
print(f"\nClassification Report on Test Set:")
print(classification_report(y_true, y_pred, target_names=['Non-Metaphor', 'Metaphor']))

# Save results to files
import json
import os

# Create results directory if it doesn't exist
os.makedirs('./results', exist_ok=True)

# Save validation results
with open('./results/validation_results.json', 'w') as f:
    json.dump(val_results, f, indent=2)

# Save test results
test_results_detailed = {
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'classification_report': classification_report(y_true, y_pred, target_names=['Non-Metaphor', 'Metaphor'], output_dict=True)
}

with open('./results/test_results.json', 'w') as f:
    json.dump(test_results_detailed, f, indent=2)

print(f"\nResults saved to ./results/")

# Save the fine-tuned model
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_model')

print(f"\nModel saved to ./final_model/")

# Example of how to reload and use the model
print(f"\nExample of how to reload the model:")
print(f"from transformers import BertForSequenceClassification, BertTokenizer")
print(f"model = BertForSequenceClassification.from_pretrained('./final_model')")
print(f"tokenizer = BertTokenizer.from_pretrained('./final_model')")