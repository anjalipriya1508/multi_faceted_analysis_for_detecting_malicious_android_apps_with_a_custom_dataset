import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Load the original dataset
df = pd.read_csv("../androZoo_dataset_analysis/overall_static_feature_extraction.csv")

# Drop non-numeric columns (like app_name or source_folder)
non_numeric_cols = df.select_dtypes(include=['object']).columns
df_numeric = df.drop(columns=non_numeric_cols)

# Fill missing values
df_numeric = df_numeric.fillna(0)

# Apply PCA with full number of components
pca = PCA()
pca.fit(df_numeric)

# Get feature importance (sum of absolute loadings across top N components)
n_components_to_consider = 10  # You can change this
loadings = np.abs(pca.components_[:n_components_to_consider])
importance_scores = loadings.sum(axis=0)

# Map importance scores to original column names
feature_importance = pd.Series(importance_scores, index=df_numeric.columns)

# Select top K original features
top_k = 25  # Set this to however many features you want to retain
top_features = feature_importance.sort_values(ascending=False).head(top_k).index.tolist()

# Build reduced DataFrame with top features + app_name (if needed)
reduced_df = df[top_features]
if "app_name" in df.columns:
    reduced_df.insert(0, "app_name", df["app_name"])

# Save the reduced original feature set
reduced_df.to_csv("../androZoo_dataset_analysis/reduced_overall_static_feature_extraction.csv", index=False)
print(f"✅ Saved reduced dataset with top {top_k} original features to 'reduced_static_features_extraction.csv'")
