import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Replace with your CSV file path
csv_path = 'androZoo_dataset_analysis/CICDataset_MalDroid_2020.csv'

# Load dataset
data = pd.read_csv(csv_path)

# Assuming the label column is named 'label'
# Features are all columns except 'label'
X = data.drop(columns=['Class']).values
y = data['Class'].values

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
labels = [1, 2, 3, 4, 5]

for i, label in enumerate(labels):
    idx = y == label
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], 
                c=colors[i], label=f'Class {label}', alpha=0.6, edgecolors='w', s=60)

plt.title("2D t-SNE plot for multiclass classification")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.show()
