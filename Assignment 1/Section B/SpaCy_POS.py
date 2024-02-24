import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Mapping from spaCy tags to dataset tags
spacy_to_dataset_tag_map = {
    'ADJ': 'JJ',
    'ADP': 'IN',
    'ADV': 'RB',
    'AUX': 'VB',
    'CONJ': 'CC',
    'CCONJ': 'CC',
    'DET': 'DT',
    'INTJ': 'UH',
    'NOUN': 'NNS',
    'NUM': 'CD',
    'PART': 'RP',
    'PRON': 'PRP',
    'PROPN': 'NNP',
    'PUNCT': ',',
    'SCONJ': 'IN',
    'SYM': 'SYM',
    'VERB': 'VB',
    'X': 'FW',
}

# Load dataset
print("Loading dataset...")
data = pd.read_csv("FinalDataset.csv", converters={"Word": eval, "POS": eval, "Tag": eval})
print("Dataset loaded.")

# Initialize containers for true and predicted tags
true_pos_tags = []
pred_pos_tags = []

for index, row in data.iterrows():
    words = row["Word"]
    true_tags = row["POS"]
    
    # Reconstruct the sentence
    sentence = " ".join(words)
    doc = nlp(sentence)
    
    # Predicted tags using spaCy with mapping to dataset tags
    pred_tags_mapped = [spacy_to_dataset_tag_map.get(token.pos_, 'UNK') for token in doc]
    
    # Since spaCy might tokenize differently, adjust the length of predicted tags
    # This step simply truncates or pads the predicted tags list to match the true tags list length
    pred_tags_mapped_adjusted = pred_tags_mapped[:len(true_tags)] + ['UNK'] * (len(true_tags) - len(pred_tags_mapped))
    
    true_pos_tags.extend(true_tags)
    pred_pos_tags.extend(pred_tags_mapped_adjusted)

# Encode tags
le = LabelEncoder()
le.fit(true_pos_tags + pred_pos_tags)
true_pos_encoded = le.transform(true_pos_tags)
pred_pos_encoded = le.transform(pred_pos_tags)

# Calculate accuracy
accuracy = accuracy_score(true_pos_encoded, pred_pos_encoded)
print(f"POS Tagging Accuracy (OurDataset): {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(true_pos_encoded, pred_pos_encoded)
# print("Confusion Matrix:")
# print(cm)

# Get class labels from the LabelEncoder
class_labels = le.classes_

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

plt.title('POS Tagging Confusion Matrix (Our Dataset)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
