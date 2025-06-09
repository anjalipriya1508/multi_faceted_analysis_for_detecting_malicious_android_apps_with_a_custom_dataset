import pandas as pd

# Load the CSV files
print("📥 Loading static features from reduced_overall_static_feature_extraction.csv ...")
static_df = pd.read_csv('androZoo_dataset_analysis/reduced_overall_static_feature_extraction.csv')

print("📥 Loading dynamic/network features from overall_dynamic_feature_extraction.csv ...")
dynamic_df = pd.read_csv('androZoo_dataset_analysis/overall_dynamic_feature_extraction.csv')

print("📥 Loading additional network analysis features from overall_pcap_feature_extraction.csv ...")
network_analysis_df = pd.read_csv('androZoo_dataset_analysis/overall_pcap_feature_extraction.csv')

# Merge static and dynamic features
print("🔗 Merging static and dynamic features on 'app_name' column ...")
merged_df = pd.merge(static_df, dynamic_df, on='app_name', how='inner')

# Drop all columns that contain 'source_folder' in their name
merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'source_folder' in col])
print(f"✅ After first merge: {len(merged_df)} records")

# Merge with network analysis features
print("🔗 Merging with network analysis features on 'app_name' column ...")
merged_df = pd.merge(merged_df, network_analysis_df, on='app_name', how='inner')

# Drop all columns that contain 'source_folder' in their name
merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'source_folder' in col])
print(f"✅ After second merge: {len(merged_df)} records")

# Drop any rows with missing values (just in case)
merged_df.dropna(inplace=True)

# Move 'label' to the last column, if it exists
if 'label' in merged_df.columns:
    label_col = merged_df.pop('label')
    merged_df['label'] = label_col

# Save final output
output_file = 'androZoo_dataset_analysis/overall_static_dynamic_network_dataset.csv'
merged_df.to_csv(output_file, index=False)

print(f"\n✅ Final merged dataset saved as: {output_file}")
print(f"📊 Total records in final dataset: {len(merged_df)}")
print(f"🧪 Total columns: {merged_df.shape[1]} — with 'label' as the last column (if present).")
