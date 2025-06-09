import os
import pandas as pd

root_dir = "androZoo_dataset_analysis"
output_file = os.path.join(root_dir, "overall_dynamic_feature_extraction.csv")

combined_data = []

# Walk through all directories
for subdir, _, files in os.walk(root_dir):
    print(f"🔍 Checking directory: {subdir}")
    for file in files:
        if file.strip() == "combined_dynamic_feature_extraction.csv":
            file_path = os.path.join(subdir, file)
            print(f"✅ Found CSV: {file_path}")
            try:
                df = pd.read_csv(file_path)
                df['source_folder'] = os.path.basename(subdir)
                combined_data.append(df)
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e.__class__.__name__} - {e}")

if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Combined CSV saved as: {output_file}")
else:
    print("\n❌ No datasets found. Make sure files are named correctly and exist in subdirectories.")
