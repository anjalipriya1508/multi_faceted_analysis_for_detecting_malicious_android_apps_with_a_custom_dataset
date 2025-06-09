import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# === Load CSV ===
csv_path = "../androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv"
df = pd.read_csv(csv_path)

print("Columns in CSV:", df.columns.tolist())
print("Unique labels in 'label' column:", df['label'].unique())

label_col = 'label'
if label_col not in df.columns:
    raise ValueError(f"CSV must contain a '{label_col}' column.")

# No mapping needed since labels are already 0 and 1
y = df[label_col].astype(int)

# Keep only numeric features, exclude the label column
X = df.drop(columns=[label_col])
X_numeric = X.select_dtypes(include=['int64', 'float64'])

if X_numeric.shape[1] < 2:
    raise ValueError("Not enough numeric features left for t-SNE after removing non-numeric columns.")

# Replace NaNs with 0
X_numeric.fillna(0, inplace=True)

# Scale features
X_scaled = StandardScaler().fit_transform(X_numeric)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create dataframe for plotting
df_tsne = pd.DataFrame({
    'x': X_tsne[:, 0],
    'y': X_tsne[:, 1],
    'label': y
})

# Create output folder
output_folder = "tsne_plots"
os.makedirs(output_folder, exist_ok=True)

# Define colors for labels
label_colors = {
    0: 'darkgreen',  # label 0
    1: 'darkred'     # label 1
}

# Plot
plt.figure(figsize=(10, 8))

for label in df_tsne['label'].unique():
    subset = df_tsne[df_tsne['label'] == label]
    color = label_colors.get(label, 'gray')
    plt.scatter(subset['x'], subset['y'], label=f"Label {label}", c=color, alpha=0.7, s=20)

plt.title("t-SNE Visualization (Label 0 vs Label 1)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_folder, "synthetic_static_dynamic_network_dataset.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ t-SNE plot saved at: {plot_path}")
