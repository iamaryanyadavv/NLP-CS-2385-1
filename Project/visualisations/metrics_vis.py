from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

model_path = '../model/IPL22_BERT_2' 
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.eval() 

# Load your test dataset
df = pd.read_csv('../data/sentiment_labeled_tweets.csv')
train_df, df_test = train_test_split(df, test_size=0.02, random_state=42)
# Preprocess text data
texts = df_test['text'].tolist()
encodings = tokenizer(texts, padding=True, truncation=True, max_length=264, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)

df_test['Sentiment'] = df_test['Sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)
true_labels = df_test['Sentiment'].tolist()

# CONFUSION MATRIX 
cm = confusion_matrix(true_labels, predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
# --------------------

# CLASSIFICATION REPORT
# Generate a classification report
report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)

# Convert the report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Plotting
fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the figure
ax.axis('tight')
ax.axis('off')
table_data = report_df.round(2)  # Round the numbers for a nicer display
the_table = ax.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc = 'center', loc='center')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.savefig('classification_report.png')  # Save the figure to a file
plt.show()  # Display the plot
# --------------------

# ROC AUC
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
roc_auc = roc_auc_score(true_labels, probabilities[:, 1])  # Assuming positive class is column 1

fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# ----------------------