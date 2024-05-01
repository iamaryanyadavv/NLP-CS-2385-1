import pandas as pd
import matplotlib.pyplot as plt

def parse_logs(file_path):
    logs = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                log_dict = eval(line)
                logs.append(log_dict)
    return logs

# Replace 'path_to_your_log_file.txt' with the actual path to your text file containing the logs.
file_path = '../model/logs/IPL22_BERT_2_logs.txt'
data = parse_logs(file_path)

# Convert list of dicts to DataFrame
df_logs = pd.DataFrame(data)

# Ensure 'epoch' is sorted if not in order
df_logs.sort_values('epoch', inplace=True)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # Adjust size as needed

# Plot loss
axs[0].plot(df_logs['epoch'], df_logs['loss'], label='Loss', color='blue')
axs[0].set_title('Loss over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot gradient norm
axs[1].plot(df_logs['epoch'], df_logs['grad_norm'], label='Gradient Norm', color='red')
axs[1].set_title('Gradient Norm over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Gradient Norm')
axs[1].legend()

# Plot learning rate
axs[2].plot(df_logs['epoch'], df_logs['learning_rate'], label='Learning Rate', color='green')
axs[2].set_title('Learning Rate over Epochs')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Learning Rate')
axs[2].legend()

# Display the plots
plt.tight_layout()
plt.show()