import pandas as pd
import os

# 👉 Replace this with your actual CSV file path
csv_file = r"androZoo_dataset_analysis/reduced_overall_static_dynamic_network_dataset.csv"

# Settings (edit if needed)
label_column = "label"       # Column that indicates malicious or benign
benign_value = 0             # Value indicating benign
malicious_value = 1          # Value indicating malicious

# Load the CSV
df = pd.read_csv(csv_file)

# Validate label column existence
if label_column not in df.columns:
    raise ValueError(f"Label column '{label_column}' not found in CSV.")

# Split the data
benign_df = df[df[label_column] == benign_value]
malicious_df = df[df[label_column] == malicious_value]

# Create output directory
output_dir = "androZoo_dataset_analysis/dataset_creation"
os.makedirs(output_dir, exist_ok=True)

# Create output filenames
base_name = os.path.splitext(os.path.basename(csv_file))[0]
benign_file = os.path.join(output_dir, f"{base_name}_benign.csv")
malicious_file = os.path.join(output_dir, f"{base_name}_malicious.csv")

# Save the split data
benign_df.to_csv(benign_file, index=False)
malicious_df.to_csv(malicious_file, index=False)

# Report
print(f"✅ Benign data saved to: {benign_file}")
print(f"⚠️ Malicious data saved to: {malicious_file}")
print(f"📊 Benign count: {len(benign_df)} | Malicious count: {len(malicious_df)}")
