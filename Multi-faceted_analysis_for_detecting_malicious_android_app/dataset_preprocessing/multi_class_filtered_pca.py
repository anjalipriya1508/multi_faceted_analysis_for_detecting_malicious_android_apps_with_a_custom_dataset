import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Load the original dataset
df = pd.read_csv("../androZoo_dataset_analysis/CICDataset_MalDroid_2020.csv")

# Identify non-numeric columns except 'Class'
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Class' in non_numeric_cols:
    non_numeric_cols.remove('Class')  # keep the Class column

# Drop non-numeric columns except 'Class'
df_numeric = df.drop(columns=non_numeric_cols)

# Fill missing values in numeric features (exclude Class)
features_only = df_numeric.drop(columns=['Class'])
features_only = features_only.fillna(0)

# Apply PCA on numeric features
pca = PCA()
pca.fit(features_only)

# Get feature importance (sum of absolute loadings across top N components)
n_components_to_consider = 25
loadings = np.abs(pca.components_[:n_components_to_consider])
importance_scores = loadings.sum(axis=0)

# Map importance scores to feature names
feature_importance = pd.Series(importance_scores, index=features_only.columns)

# Select top K features
top_k = 10
top_features = feature_importance.sort_values(ascending=False).head(top_k).index.tolist()

# Create reduced DataFrame with top features + Class column
reduced_df = df_numeric[top_features + ['Class']]

# Save reduced dataset
reduced_df.to_csv("../androZoo_dataset_analysis/reduced_CICDataset_MalDroid_2020.csv", index=False)
print(f"✅ Saved reduced dataset with top {top_k} features + 'Class' column to 'reduced_CICDataset_MalDroid_2020.csv'")
