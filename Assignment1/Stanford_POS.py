import pandas as pd
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk

# Ensure punkt tokenizer models are pre-downloaded
nltk.download('punkt', quiet=True)

# Set the Java path, if not already set in the environment variables
java_path = "/usr/bin/java"  # Adjust this to the path of your Java executable
os.environ['JAVAHOME'] = java_path

# Replace these paths with the correct paths to your Stanford POS tagger and model
stanford_pos_jar = '/Users/admin/Downloads/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_pos_model = '/Users/admin/Downloads/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'

# Initialize the Stanford POS tagger
st_tagger = StanfordPOSTagger(model_filename=stanford_pos_model, path_to_jar=stanford_pos_jar)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("/Users/admin/NLP-CS-2385-1/Assignment1/NER_Dataset.csv", converters={"Word": eval, "POS": eval, "Tag": eval})
print("Dataset loaded.")

# Prepare sentences for batch tagging
sentences_for_tagging = [word_tokenize(" ".join(row['Word'])) for index, row in data.iterrows()]

# Tag all sentences at once
all_tagged_sentences = st_tagger.tag_sents(sentences_for_tagging)

print("1")
# Initialize containers for true and predicted tags
true_pos_tags = []
pred_pos_tags = []

# Extract predicted tags and align them with true tags
for row, tagged_sentence in zip(data.iterrows(), all_tagged_sentences):
    index, row = row
    true_tags = row['POS']
    pred_tags = [tag for _, tag in tagged_sentence]
    
    # Adjust the length of the predicted tags list to match the true tags list
    pred_tags_adjusted = pred_tags[:len(true_tags)] + ['UNK'] * (len(true_tags) - len(pred_tags))
    
    true_pos_tags.extend(true_tags)
    pred_pos_tags.extend(pred_tags_adjusted)
# Encode the tags using the LabelEncoder
label_encoder = LabelEncoder()

# Combine true and predicted tags and then fit the LabelEncoder to ensure all labels are known
combined_tags = true_pos_tags + pred_pos_tags
label_encoder.fit(combined_tags)

# Now transform both true and predicted tags
true_encoded = label_encoder.transform(true_pos_tags)
pred_encoded = label_encoder.transform(pred_pos_tags)

# Calculate accuracy
accuracy = accuracy_score(true_encoded, pred_encoded)
print(f"POS Tagging Accuracy with Stanford POS Tagger: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(true_encoded, pred_encoded)

# Plotting the confusion matrix
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Stanford POS Tagger')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()



#POS Tagging Accuracy with Stanford POS Tagger: 0.9578