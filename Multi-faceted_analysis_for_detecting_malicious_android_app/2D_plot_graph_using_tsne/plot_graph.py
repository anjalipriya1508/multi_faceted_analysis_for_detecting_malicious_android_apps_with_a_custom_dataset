import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# === Load CSV ===
csv_path = "../androZoo_dataset_analysis/TUANDROMD.csv"  # <-- Replace with your actual file
df = pd.read_csv(csv_path)

# === Check and separate label ===
label_col = 'Label'
if label_col not in df.columns:
    raise ValueError("CSV must contain a 'Label' column.")

# === Keep only numeric features ===
X = df.drop(columns=[label_col])
X_numeric = X.select_dtypes(include=['int64', 'float64'])

if X_numeric.shape[1] < 2:
    raise ValueError("Not enough numeric features left for t-SNE after removing non-numeric columns.")

# === Replace NaNs with 0 ===
X_numeric.fillna(0, inplace=True)

# === Get label column ===
y = df[label_col].astype(str)  # Convert labels to string for safety

# === Scale and apply t-SNE ===
X_scaled = StandardScaler().fit_transform(X_numeric)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# === Create t-SNE dataframe ===
df_tsne = pd.DataFrame({
    'x': X_tsne[:, 0],
    'y': X_tsne[:, 1],
    'Label': y
})

# === Create output folder ===
output_folder = "tsne_plots"
os.makedirs(output_folder, exist_ok=True)

# === Define label colors ===
label_colors = {
    'B': 'darkgreen',
    'S': 'darkred'
}

# === Plotting ===
plt.figure(figsize=(10, 8))

for label in df_tsne['Label'].unique():
    subset = df_tsne[df_tsne['Label'] == label]
    color = label_colors.get(label.lower(), 'gray')
    plt.scatter(subset['x'], subset['y'], label=str(label).capitalize(), c=color, alpha=0.7, s=20)

plt.title("t-SNE Visualization (Begnin vs Malicious)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save plot ===
plot_path = os.path.join(output_folder, "tsne_begnin_vs_malicious.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ t-SNE plot saved at: {plot_path}")
