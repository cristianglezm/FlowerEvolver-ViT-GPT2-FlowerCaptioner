import json
import os
import matplotlib.pyplot as plt
from collections import Counter

# Path to the folder containing JSON files
folder_path = './data/genomes'

# Initialize counters for petals attributes
bias_counter = Counter()
num_layers_counter = Counter()
P_counter = Counter()

# Load and evaluate JSON files
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        with open(os.path.join(folder_path, file_name)) as f:
            data = json.load(f)
            petals = data.get('Flower', {}).get('petals', {})
            bias_counter[petals.get('bias')] += 1
            num_layers_counter[petals.get('numLayers')] += 1
            P_counter[petals.get('P')] += 1

# Plotting the distribution
plt.figure(figsize=(12, 8))

# Bias plot
plt.subplot(3, 1, 1)
plt.bar(bias_counter.keys(), bias_counter.values())
plt.title('Distribution of Bias')
plt.xlabel('Bias')
plt.ylabel('Count')

# numLayers plot
plt.subplot(3, 1, 2)
plt.bar(num_layers_counter.keys(), num_layers_counter.values())
plt.title('Distribution of numLayers')
plt.xlabel('Number of Layers')
plt.ylabel('Count')

# P plot
plt.subplot(3, 1, 3)
plt.bar(P_counter.keys(), P_counter.values())
plt.title('Distribution of P')
plt.xlabel('P')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
