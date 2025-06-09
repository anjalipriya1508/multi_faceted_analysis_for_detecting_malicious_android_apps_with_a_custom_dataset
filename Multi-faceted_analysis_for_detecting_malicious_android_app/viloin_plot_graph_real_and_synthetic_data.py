import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import textwrap

# === Load the datasets ===
real_data = pd.read_csv("androZoo_dataset_analysis/reduced_overall_static_dynamic_network_dataset.csv").iloc[:, 1:]
synthetic_data = pd.read_csv("androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv").iloc[:, 1:]

# === Add dataset source labels ===
real_data['source'] = 'Real'
synthetic_data['source'] = 'Synthetic'
combined = pd.concat([real_data, synthetic_data], ignore_index=True)

# === Identify feature types ===
features = [col for col in combined.columns if col != 'source']
discrete_features = []
continuous_features = []

for col in features:
    if pd.api.types.is_numeric_dtype(combined[col]):
        if combined[col].nunique() <= 10:
            discrete_features.append(col)
        else:
            continuous_features.append(col)

# === Create directories for saving images ===
barplot_dir = "barplot_images"
violinplot_dir = "violinplot_images"
os.makedirs(barplot_dir, exist_ok=True)
os.makedirs(violinplot_dir, exist_ok=True)

# === Prepare wrapped feature names for barplot ===
def wrap_labels(labels, width=20):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

wrapped_discrete_features = wrap_labels(discrete_features, width=42)

# === Horizontal Bar plot for discrete features ===
mean_data = combined.groupby("source")[discrete_features].mean().transpose().reset_index()
mean_data.rename(columns={"index": "Feature"}, inplace=True)
melted = pd.melt(mean_data, id_vars="Feature", var_name="Source", value_name="Mean")

feature_label_map = dict(zip(discrete_features, wrapped_discrete_features))
melted['Feature_wrapped'] = melted['Feature'].map(feature_label_map)

plt.figure(figsize=(20, max(12, len(discrete_features) * 0.7)))

sns.barplot(
    data=melted,
    y="Feature_wrapped",
    x="Mean",
    hue="Source",
    palette={"Real": "skyblue", "Synthetic": "lightcoral"},
    orient='h',
    dodge=True
)

plt.xlabel("Mean Value", fontsize=16, fontweight='bold')
plt.ylabel("Feature", fontsize=18, fontweight='bold')
plt.title("Mean Values of Discrete Features: Real vs Synthetic", fontsize=20, fontweight='bold')
plt.grid(axis='x')
plt.legend(title="Dataset", loc='best', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.3)

barplot_path = os.path.join(barplot_dir, "discrete_features_barplot.png")
plt.savefig(barplot_path, dpi=300)
plt.close()
print(f"Bar plot saved as image: {barplot_path}")

# === Save combined split-violin plots in batches of 4 ===
batch_size = 4
num_batches = math.ceil(len(continuous_features) / batch_size)

for batch_idx in range(num_batches):
    batch_features = continuous_features[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns
    axs = axs.flatten()

    for i, feature in enumerate(batch_features):
        # Melt the data to use feature as x, value as y, source as hue
        plot_data = combined[['source', feature]].copy()
        plot_data = plot_data.rename(columns={feature: "value"})
        plot_data["Feature"] = feature

        sns.violinplot(
            x="Feature",
            y="value",
            hue="source",
            data=plot_data,
            split=True,
            inner='quartile',
            palette={"Real": "skyblue", "Synthetic": "lightcoral"},
            ax=axs[i]
        )

        axs[i].set_title(f'Split Violin - {feature}', fontsize=14, fontweight='bold')
        axs[i].set_xlabel('')
        axs[i].set_xticks([]) 
        axs[i].set_ylabel(feature, fontsize=12, fontweight='bold')
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(True)
        axs[i].legend_.remove()  # Remove legend from individual plots

    # Remove empty axes if batch has fewer than 4 plots
    for j in range(len(batch_features), 4):
        fig.delaxes(axs[j])

    # Add shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc='upper right', fontsize=12)

    plt.tight_layout()
    violin_path = os.path.join(violinplot_dir, f"violinplots_split_batch_{batch_idx+1}.png")
    plt.savefig(violin_path, dpi=300)
    plt.close()
    print(f"Saved split violin plots batch {batch_idx+1} as image: {violin_path}")
