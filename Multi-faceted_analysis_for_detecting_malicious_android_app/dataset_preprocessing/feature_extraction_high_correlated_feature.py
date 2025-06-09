import pandas as pd
import numpy as np

# Load dataset
df_full = pd.read_csv("../androZoo_dataset_analysis/overall_static_dynamic_network_dataset.csv")

# Separate identifier and target
app_names = df_full["app_name"]
label = df_full["label"]
features = df_full.drop(columns=["app_name", "label"])

# Step 1️⃣: Select features correlated with the label
label_corr = features.corrwith(label).abs()
label_corr_threshold = 0.3  # You can adjust this
relevant_features = label_corr[label_corr >= label_corr_threshold].index.tolist()
filtered_features = features[relevant_features]

print(f"✅ Step 1: {len(relevant_features)} features selected with |correlation to label| ≥ {label_corr_threshold}")

# 🛡️ Remove features with zero variance (to prevent NaNs in correlation matrix)
filtered_features = filtered_features.loc[:, filtered_features.nunique() > 1]

# Step 2️⃣: Remove highly redundant features
corr_matrix = filtered_features.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

redundancy_threshold = 0.9  # You can adjust this
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > redundancy_threshold)]
final_features = filtered_features.drop(columns=to_drop)

print(f"✅ Step 2: {len(to_drop)} redundant features removed (correlation > {redundancy_threshold})")

# Assemble final dataset
df_final = pd.concat([app_names, final_features, label], axis=1)
df_final.to_csv("../androZoo_dataset_analysis/reduced_overall_static_dynamic_network_dataset.csv", index=False)

# Summary
print(f"\n📊 Total original features: {features.shape[1]}")
print(f"🧠 Final features retained: {final_features.shape[1]}")
print(f"📁 Output file: reduced_overall_static_dynamic_network_dataset.csv")
