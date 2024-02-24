import pandas as pd
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# nltk.download('punkt')

# Replace these paths with the correct paths to your Stanford POS tagger and model
stanford_pos_jar = 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'
stanford_pos_model = 'stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
# stanford_pos_model = 'stanford-postagger-full-2020-11-17/models/english-caseless-left3words-distsim.tagger'

# Initialize the Stanford POS tagger
st_tagger = StanfordPOSTagger(model_filename=stanford_pos_model, path_to_jar=stanford_pos_jar)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("NER_Dataset.csv", converters={"Word": eval, "POS": eval, "Tag": eval})
print("Dataset loaded.")

# Initialize containers for true and predicted tags
true_pos_tags = []
pred_pos_tags = []

# Process the dataset and tag with Stanford POS tagger
for index, row in data.iterrows():
    sentence = " ".join(row['Word'])
    tokenized_sentence = word_tokenize(sentence)
    print(index)
    # if index<20:
    #Tag the tokenized sentence
    tagged_sentence = st_tagger.tag(tokenized_sentence)
    
    # Extract the predicted tags
    pred_tags = [tag for _, tag in tagged_sentence]
    
    # Since the tokenization might differ, adjust the length of the predicted tags list
    pred_tags_adjusted = pred_tags[:len(row['POS'])] + ['UNK'] * (len(row['POS']) - len(pred_tags))
    
    # Append the true and predicted tags to their respective lists
    true_pos_tags.extend(row['POS'])
    pred_pos_tags.extend(pred_tags_adjusted)

# Encode the tags using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(true_pos_tags + pred_pos_tags)
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
