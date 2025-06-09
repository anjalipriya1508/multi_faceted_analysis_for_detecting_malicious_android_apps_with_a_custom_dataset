import pandas as pd

# Load the CSV files
print("📥 Loading static features from combined_static_feature_extraction.csv ...")
static_df = pd.read_csv('androZoo_dataset_analysis/combined_static_feature_extraction.csv')

print("📥 Loading dynamic/network features from combined_dynamic_feature_extraction.csv ...")
dynamic_df = pd.read_csv('androZoo_dataset_analysis/combined_dynamic_feature_extraction.csv')

print("📥 Loading additional network analysis features from combined_pcap_feature_extraction.csv ...")
network_analysis_df = pd.read_csv('androZoo_dataset_analysis/combined_pcap_feature_extraction.csv')

# Merge static and dynamic features
print("🔗 Merging static and dynamic features on 'app_name' column ...")
merged_df = pd.merge(static_df, dynamic_df, on='app_name', how='inner')

# Merge with network analysis features
print("🔗 Merging with network analysis features on 'app_name' column ...")
merged_df = pd.merge(merged_df, network_analysis_df, on='app_name', how='inner')

# Ensure 'label' is moved to the last column
if 'label' in merged_df.columns:
    label_col = merged_df.pop('label')  # Remove label column
    merged_df['label'] = label_col      # Append it to the end

# Save the merged result
output_file = 'androZoo_dataset_analysis/overall_static_dynamic_network_dataset.csv'
merged_df.to_csv(output_file, index=False)

print(f"\n✅ Merged dataset successfully saved as {output_file}")
print(f"📊 Total records in merged dataset: {len(merged_df)}")
print(f"🧪 Total columns: {merged_df.shape[1]} — with 'label' as the last column.")
