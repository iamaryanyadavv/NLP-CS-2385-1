import pandas as pd
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure punkt tokenizer models are pre-downloaded
nltk.download('punkt', quiet=True)

# Set the Java path, if not already set in the environment variables
java_path = "/usr/bin/java"  # Adjust this to the path of your Java executable
os.environ['JAVAHOME'] = java_path

# Replace these paths with the correct paths to your Stanford NER tagger and model
stanford_ner_jar = '/Users/admin/Downloads/stanford-ner-2020-11-17/stanford-ner.jar'
stanford_ner_model = '/Users/admin/Downloads/stanford-ner-2020-11-17/classifiers/english.muc.7class.distsim.crf.ser.gz'

# Initialize the Stanford NER tagger
st_ner_tagger = StanfordNERTagger(model_filename=stanford_ner_model, path_to_jar=stanford_ner_jar)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("/Users/admin/NLP-CS-2385-1/Assignment1/FinalDataset.csv", converters={"Word": eval, "Tag": eval})
print("Dataset loaded.")

# Prepare sentences for batch tagging
sentences_for_tagging = [word_tokenize(" ".join(row['Word'])) for index, row in data.iterrows()]

# Tag all sentences at once using the NER tagger
all_tagged_sentences = st_ner_tagger.tag_sents(sentences_for_tagging)

# Initialize containers for true and predicted NER tags
true_ner_tags = []
pred_ner_tags = []

# Extract predicted NER tags and align them with true NER tags
for row, tagged_sentence in zip(data.iterrows(), all_tagged_sentences):
    index, row = row
    true_tags = row['Tag']
    pred_tags = [ner_tag for _, ner_tag in tagged_sentence]

    # Adjust the length of the predicted NER tags list to match the true tags list
    pred_tags_adjusted = pred_tags[:len(true_tags)] + ['O'] * (len(true_tags) - len(pred_tags))

    true_ner_tags.extend(true_tags)
    pred_ner_tags.extend(pred_tags_adjusted)

# Encode the NER tags using the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(true_ner_tags + pred_ner_tags)  # Fit to all possible labels

true_encoded = label_encoder.transform(true_ner_tags)
pred_encoded = label_encoder.transform(pred_ner_tags)

accuracy = accuracy_score(true_encoded, pred_encoded)
print(f"POS Tagging Accuracy with Stanford POS Tagger: {accuracy:.4f}")
# Generate confusion matrix
cm = confusion_matrix(true_encoded, pred_encoded, labels=label_encoder.transform(label_encoder.classes_))

# Plotting the confusion matrix
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Stanford NER Tagger')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

#POS Tagging Accuracy with Stanford POS Tagger: 0.7754