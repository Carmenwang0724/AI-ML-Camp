from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./final_model')
tokenizer = BertTokenizer.from_pretrained('./final_model')

# Test sentence (this is a known metaphor)
test_sentence = "The world is a stage."

# Tokenize the sentence
inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Get predictions from the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()  # 0 = not a metaphor, 1 = metaphor

# Print raw output and predicted class
print(f"Test Sentence: {test_sentence}")
print(f"Predicted class: {predicted_class} (0 = not a metaphor, 1 = metaphor)")
