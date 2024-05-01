import pandas as pd
from transformers import pipeline
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Load the dataset
df = pd.read_csv('filtered_and_cleaned_tweets.csv')

# Load a pre-trained sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

# Function to apply model and extract sentiment
def get_sentiment(text):
    result = classifier(text)
    return result[0]['label']

# # Limit the DataFrame to the first 20 rows
# df_subset = df.head(20)

# Apply the sentiment analysis
df['Sentiment'] = df['text'].apply(get_sentiment)

# Export the DataFrame with sentiment labels to a new CSV file
df.to_csv('sentiment_labeled_tweets.csv', index=False)