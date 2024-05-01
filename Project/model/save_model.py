from transformers import BertForSequenceClassification

# Path to your checkpoint directory
checkpoint_path = "./results/checkpoint-7431"

# Load the model from checkpoint
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# Save the model to disk
model.save_pretrained('./IPL22_BERT_2')
print('Model saved from checkpoint.')
