import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from transformers import TrainerCallback

print('Reading dataset')
# Load the dataset
df = pd.read_csv('../data/sentiment_labeled_tweets.csv')
df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

print('Read\n')

# Check the distribution of sentiment
distribution = df['Sentiment'].value_counts()
# print(distribution)

# Optionally, calculate the percentage distribution
percentage_distribution = df['Sentiment'].value_counts(normalize=True) * 100
# print(percentage_distribution)

import matplotlib.pyplot as plt

# Plot the distribution of sentiments
distribution.plot(kind='bar', color=['blue', 'green'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)  # Adjust labels based on your data
plt.show()



print('BERT model init')
# Split the dataset into training and testing sets
train_df, eval_df = train_test_split(df, test_size=0.5, random_state=42)
print(f'Training on {len(train_df)} rows')

# Tokenizer and dataset preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TweetsDataset(Dataset):
    def __init__(self, df):
        self.labels = df['Sentiment'].tolist()
        self.texts = [tokenizer(text, 
                                padding='max_length',  # Pad sequences to a consistent length
                                max_length=264,  # Maximum length of tokens
                                truncation=True,  # Truncate texts longer than max_length
                                return_tensors="pt")  # Return PyTorch tensors
                      for text in df['text']]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val.squeeze() for key, val in self.texts[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, grad_norm_threshold=0.1):
        self.grad_norm_threshold = grad_norm_threshold
        self.grad_norm = None  # To capture grad_norm across steps if not directly available

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This method is called whenever the log method is called, which is where logs would be available
        if logs is not None and 'grad_norm' in logs:
            self.grad_norm = logs['grad_norm']

    def on_step_end(self, args, state, control, **kwargs):
        # Check the grad_norm captured during logging
        if self.grad_norm is not None and self.grad_norm < self.grad_norm_threshold:
            print(f"Stopping early, gradient norm {self.grad_norm} below threshold {self.grad_norm_threshold}")
            control.should_training_stop = True

    
train_dataset = TweetsDataset(train_df)
eval_dataset = TweetsDataset(eval_df)

# Model preparation
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=2,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=100,
#     learning_rate=5e-5,
#     evaluation_strategy="no",  # Do not evaluate during training
#     save_strategy="epoch",  # Save only at the end of training
#     load_best_model_at_end=False,  # Do not try to load the best model automatically
#     metric_for_best_model="accuracy",
#     # fp16=True  # Enable mixed precision
# )

# Modify TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    evaluation_strategy="epoch", 
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="all"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(grad_norm_threshold=0.05)]  # You can adjust the threshold
)
print('BERT model loaded\n')

print('Training the model...')
# Train the model
trainer.train()
print('Trained.\n')

# Evaluate the model
results = trainer.evaluate()
print(results)

print('Saving model to disk...')
model.save_pretrained('./IPL22_BERT_2')
print('Saved.\n')