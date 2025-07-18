import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# 1. Text Preprocessing (Remove numbers and unwanted symbols)
def clean_text(text):
    """Clean text by removing unnecessary spaces, special symbols, and numbers."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^\w\s.,!?¿¡]', '', text)  # Remove special characters (optional)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# 2. Load and Clean the Text from Files
file_paths = {
    '/Users/Admin/AI-ML-Camp/AI-ML-Camp/Memorias_full_text.txt': 'Memorias',
    '/Users/Admin/AI-ML-Camp/AI-ML-Camp/Vita_full_text.txt': 'Vita',
    '/Users/Admin/AI-ML-Camp/AI-ML-Camp/arboleda_full_text.txt': 'Arboleda'
}

all_sentences = []
file_sources = []

for file_path, label in file_paths.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Split the text into sentences (naive split by periods, can be improved)
    sentences = cleaned_text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences

    # Add non-empty sentences and their simplified source label
    all_sentences.extend(sentences)  # Collect all sentences
    file_sources.extend([label] * len(sentences))  # Record the simplified label for each sentence

# # 3. Tokenize the Text
# tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

# 4. Load the Pretrained Model
model = BertForSequenceClassification.from_pretrained('./final_model') 
tokenizer = BertTokenizer.from_pretrained('./final_model')

# 5. Perform Inference for Each Sentence
metaphor_count = 0
metaphor_sentences = []
file_column = []

for sentence, file_label in zip(all_sentences, file_sources):
    # Tokenize the sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Perform inference with no gradient tracking (for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()  # Get the predicted class (0 or 1)

    # If the sentence is a metaphor (class = 1), save and print it
    if prediction == 1:
        metaphor_count += 1
        metaphor_sentences.append(sentence)
        file_column.append(file_label)  # Save the simplified source label for the metaphor sentence

# 6. Output Results
print(f"Number of metaphors detected: {metaphor_count}\n")
print("Sentences with metaphors:")
for sentence, file in zip(metaphor_sentences, file_column):
    print(f"Source: {file}, Sentence: {sentence}")

# 7. Save Results to CSV
# Create a DataFrame with the metaphor sentences and their simplified sources
df = pd.DataFrame({
    "Metaphor Sentence": metaphor_sentences,
    "Source File": file_column
})

# Save to CSV
output_csv_path = "/Users/Admin/AI-ML-Camp/AI-ML-Camp/metaphor_sentences_by_file.csv"
df.to_csv(output_csv_path, index=False)

print(f"\nThe metaphor sentences have been saved to {output_csv_path}")
