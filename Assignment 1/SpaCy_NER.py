import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
print('Loading dataset.')
data = pd.read_csv('NER_Dataset.csv', converters={'Word': eval, 'POS': eval, 'Tag': eval})
print('Dataset loaded.')

# Containers for ground truth and predictions
true_pos_tags = []
pred_pos_tags = []

true_ner_tags = []
pred_ner_tags = []

print('Applying models on dataset.')
for index, row in data.iterrows():
    # Reconstruct the sentence
    sentence = " ".join(row['Word'])
    doc = nlp(sentence)

    spacy_ner_tags = ['O' for _ in doc]  # Initialize with 'O'
    for ent in doc.ents:
        spacy_ner_tags[ent.start] = f'B-{ent.label_}'
        for i in range(ent.start + 1, ent.end):
            spacy_ner_tags[i] = f'I-{ent.label_}'

    aligned_ner_tags = spacy_ner_tags[:len(row['Word'])]

    true_ner_tags.extend(row['Tag'])
    pred_ner_tags.extend(aligned_ner_tags)

print('Models applied successfully.')

# Encoding and transformation
le = LabelEncoder()

# Fit and transform true and predicted labels for NER
le.fit(true_ner_tags + pred_ner_tags)
true_ner_encoded = le.transform(true_ner_tags)
pred_ner_encoded = le.transform(pred_ner_tags)

# Calculate and print accuracy for POS and NER
ner_accuracy = accuracy_score(true_ner_encoded, pred_ner_encoded)
print(f"NER Tagging Accuracy: {ner_accuracy}")

# Generate and print confusion matrices
labels_ner = np.unique(np.concatenate((true_ner_encoded, pred_ner_encoded)))
cm= confusion_matrix(true_ner_encoded, pred_ner_encoded, labels=labels_ner)

# Get class labels from the LabelEncoder
class_labels = le.classes_

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

plt.title('NER Tagging Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

