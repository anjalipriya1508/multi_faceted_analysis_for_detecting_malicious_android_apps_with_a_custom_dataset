import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
from ctgan import CTGAN

# Load dataset
data = pd.read_csv('androZoo_dataset_analysis/dataset_creation/reduced_overall_static_dynamic_network_dataset_malicious.csv')
print("✅ Loaded Dataset:")
print(data.head())

# Basic info
print("\n🧼 Data Info:")
print(data.info())
print("\n🧼 Null counts:")
print(data.isnull().sum())

# Detect categorical and binary columns
auto_categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
manual_categorical_columns = ['app_name'] if 'app_name' in data.columns else []
categorical_columns = list(set(auto_categorical_columns + manual_categorical_columns))

binary_columns = [
    col for col in data.columns
    if data[col].nunique() == 2 and sorted(data[col].dropna().unique()) == [0, 1]
]
if 'label' in binary_columns:
    binary_columns.remove('label')
categorical_columns = list(set(categorical_columns + binary_columns))

# Identify numeric columns
numeric_columns = [col for col in data.columns if col not in categorical_columns and col != 'label']

print("\n📋 Final Categorical Columns:", categorical_columns)
print("\n🔢 Numeric Columns:", numeric_columns)

# Train CTGAN
print(f"\n🧠 Training CTGAN on {len(data)} samples with {len(data.columns)} features...")
ctgan = CTGAN(epochs=5000, verbose=True)
ctgan.fit(data, categorical_columns)

# Generate synthetic data
num_samples = 5000
print(f"\n🧪 Generating {num_samples} synthetic samples...")
synthetic_data = ctgan.sample(num_samples)

# Show preview
print("\n🧾 Sample of Synthetic Data:")
print(synthetic_data.head())

# Clean numeric columns: convert to numeric, fill NaNs with mean of original data
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    synthetic_data[col] = pd.to_numeric(synthetic_data[col], errors='coerce')
    mean_val = data[col].mean()
    data[col].fillna(mean_val, inplace=True)
    synthetic_data[col].fillna(mean_val, inplace=True)
    synthetic_data[col] = synthetic_data[col].clip(lower=0)

# Save full synthetic data
synthetic_data.to_csv('androZoo_dataset_analysis/dataset_creation/malicious_synthetic_data_full.csv', index=False)
print("💾 Saved full synthetic data to 'synthetic_data_full.csv'")

# Downsample to original dataset size
synthetic_sampled = synthetic_data.sample(n=len(data), random_state=42).reset_index(drop=True)
synthetic_sampled.to_csv('androZoo_dataset_analysis/dataset_creation/malicious_synthetic_data_604.csv', index=False)
print("💾 Saved downsampled synthetic data to 'synthetic_data_604.csv'")

# Distribution check
print("\n🔍 Checking numeric column similarity:")
for col in numeric_columns[:3]:
    print(f"{col}: Original mean = {data[col].mean():.4f}, Synthetic mean = {synthetic_data[col].mean():.4f}")

print("\n🔍 Label Distribution (Original vs Synthetic):")
print("Original:\n", data['label'].value_counts())
print("Synthetic:\n", synthetic_data['label'].value_counts())

print("\n🔍 Comparing first few rows of original vs synthetic data:\n")
print("Original:\n", data.head(), "\n")
print("Synthetic:\n", synthetic_data.head(), "\n")

# Plot distributions
def plot_feature_distribution(data_orig, data_synth, column, categorical_columns):
    plt.figure(figsize=(10, 6))
    if column in categorical_columns:
        combined = pd.concat([
            data_orig[[column]].assign(dataset='Original'),
            data_synth[[column]].assign(dataset='Synthetic')
        ])
        sns.countplot(data=combined, x=column, hue='dataset', palette=['blue', 'orange'])
    else:
        if data_orig[column].nunique() <= 1 and data_synth[column].nunique() <= 1:
            print(f"⚠️ Skipping '{column}': No variance in both datasets.")
            return
        sns.kdeplot(data_orig[column], label='Original', fill=True, color='blue', warn_singular=False)
        sns.kdeplot(data_synth[column], label='Synthetic', fill=True, color='orange', warn_singular=False)
        stat, p_val = ks_2samp(data_orig[column], data_synth[column])
        plt.title(f'Distribution Comparison: {column}\nKS-stat={stat:.4f}, p={p_val:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n📊 Visualizing Categorical Features:")
for col in categorical_columns[:3]:
    plot_feature_distribution(data, synthetic_sampled, col, categorical_columns)

print("\n📊 Visualizing Continuous Features:")
for col in numeric_columns[:3]:
    plot_feature_distribution(data, synthetic_sampled, col, categorical_columns)

# t-SNE Visualization
print("\n📉 Performing t-SNE 2D visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
combined_data = pd.concat([
    data[numeric_columns].reset_index(drop=True),
    synthetic_sampled[numeric_columns].reset_index(drop=True)
])
tsne_result = tsne.fit_transform(combined_data)

tsne_df = pd.DataFrame(tsne_result, columns=["TSNE-1", "TSNE-2"])
tsne_df["Source"] = ['Original'] * len(data) + ['Synthetic'] * len(synthetic_sampled)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=tsne_df, x="TSNE-1", y="TSNE-2", hue="Source", palette=['blue', 'orange'])
plt.title("t-SNE Projection of Original vs Synthetic Samples")
plt.tight_layout()
plt.show()
