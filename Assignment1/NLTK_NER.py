import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to get NER tags from NLTK's tree
def get_nltk_ner_tags(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    ne_tree = ne_chunk(pos_tags)
    
    ner_tags = []
    for i in ne_tree:
        if hasattr(i, 'label'):
            label = i.label()
            ner_tags.append(f'B-{label}')
            for _ in range(1, len(i)):
                ner_tags.append(f'I-{label}')
        else:
            ner_tags.append('O')
    return ner_tags

# Adjust the path to where your dataset is located
print('Loading dataset.')
data = pd.read_csv('/Users/admin/NLP-CS-2385-1/Assignment 1/NER_Dataset.csv', converters={'Word': eval, 'POS': eval, 'Tag': eval})
print('Dataset loaded.')

true_ner_tags = []
pred_ner_tags = []

print('Applying models on dataset.')
for index, row in data.iterrows():
    sentence = " ".join(row['Word'])
    
    # Get NLTK NER tags
    nltk_ner_tags = get_nltk_ner_tags(sentence)
    
    # Ensure the predicted tags list matches the actual number of words
    if len(nltk_ner_tags) < len(row['Word']):
        # If fewer predicted tags, pad with 'O'
        nltk_ner_tags += ['O'] * (len(row['Word']) - len(nltk_ner_tags))
    elif len(nltk_ner_tags) > len(row['Word']):
        # If more predicted tags, truncate the list
        nltk_ner_tags = nltk_ner_tags[:len(row['Word'])]

    pred_ner_tags.extend(nltk_ner_tags)
    true_ner_tags.extend(row['Tag'])

print('Models applied successfully.')

# Encoding and transformation
le = LabelEncoder()
le.fit(true_ner_tags + pred_ner_tags)
true_ner_encoded = le.transform(true_ner_tags)
pred_ner_encoded = le.transform(pred_ner_tags)

# Calculate and print accuracy for NER
ner_accuracy = accuracy_score(true_ner_encoded, pred_ner_encoded)
print(f"NER Tagging Accuracy: {ner_accuracy}")

# Generate and print confusion matrix
labels_ner = np.unique(np.concatenate((true_ner_encoded, pred_ner_encoded)))
cm = confusion_matrix(true_ner_encoded, pred_ner_encoded, labels=labels_ner)

# Get class labels from the LabelEncoder
class_labels = le.classes_

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('NER Tagging Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#NER Tagging Accuracy: 0.8392361061440526