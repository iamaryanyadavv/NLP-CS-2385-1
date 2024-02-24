import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load dataset
print("Loading dataset...")
data = pd.read_csv("/Users/admin/NLP-CS-2385-1/Assignment 1/Section B/FinalDataset.csv", converters={"Word": eval, "POS": eval, "Tag": eval})
print("Dataset loaded.")

# Initialize containers for true and predicted tags
true_pos_tags = []
pred_pos_tags = []

for index, row in data.iterrows():
    words = row["Word"]
    true_tags = row["POS"]
    
    # Tokenize the sentence (assuming 'words' are already tokenized correctly)
    tokens = words  # Use word_tokenize(sentence) if 'words' needs tokenization

    # Get POS tags for the tokens using NLTK
    tagged = nltk.pos_tag(tokens)
    
    # Extract predicted tags
    pred_tags = [tag for _, tag in tagged]
    
    # Since NLTK's tokenization and tagging should align with 'words', direct comparison is possible
    true_pos_tags.extend(true_tags)
    pred_pos_tags.extend(pred_tags)

# Encode tags to handle them as numerical data for confusion matrix
le = LabelEncoder()
le.fit(true_pos_tags + pred_pos_tags)
true_pos_encoded = le.transform(true_pos_tags)
pred_pos_encoded = le.transform(pred_pos_tags)

# Calculate accuracy
accuracy = accuracy_score(true_pos_encoded, pred_pos_encoded)
print(f"POS Tagging Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(true_pos_encoded, pred_pos_encoded)

# Get class labels from the LabelEncoder for plotting
class_labels = le.classes_

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

plt.title('POS Tagging Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


#POS Tagging Accuracy: 0.9589