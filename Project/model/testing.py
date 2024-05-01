import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model
model_path = './IPL22_BERT_2'
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Model and tokenizer loaded.")

def preprocess(text):
    # Tokenize the text and prepare the format that the model expects
    inputs = tokenizer(text, padding='max_length', max_length=264, truncation=True, return_tensors="pt")
    return inputs


def predict_sentiment(text):
    # Preprocess the text
    inputs = preprocess(text)

    # Predict
    with torch.no_grad():  # No need to compute gradients when making predictions
        outputs = model(**inputs)
    
    # Get the prediction probabilities
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convert these probabilities to binary predictions
    predicted_class = torch.argmax(predictions).item()
    
    # Map the predicted class to sentiment labels
    labels = ['Negative', 'Positive']
    sentiment = labels[predicted_class]
    return sentiment

test_texts = [
    "Ruturaj is a good player, I like him a lot",
    "Ruturaj is not as good a player as I thought he was"
]

# Test each text
for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

