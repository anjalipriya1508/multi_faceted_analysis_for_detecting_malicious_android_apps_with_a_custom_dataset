import os
import pandas as pd

root_dir = "androZoo_dataset_analysis"
output_file = os.path.join(root_dir, "overall_static_feature_extraction.csv")

combined_data = []
all_columns = set()

# First pass: collect all columns
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.strip() == "combined_static_feature_extraction.csv":
            file_path = os.path.join(subdir, file)
            try:
                df = pd.read_csv(file_path)
                all_columns.update(df.columns.tolist())
            except Exception as e:
                print(f"❌ Failed to read {file_path} in first pass: {e.__class__.__name__} - {e}")

# Ensure 'app_name' exists
if "app_name" not in all_columns:
    print("❌ 'app_name' column missing in input CSVs.")
    exit()

# Prepare ordered columns (app_name first, then sorted others, source_folder last)
all_columns.discard("app_name")
ordered_columns = ["app_name"] + sorted(all_columns)

# Second pass: reindex and fill missing values
for subdir, _, files in os.walk(root_dir):
    print(f"🔍 Checking directory: {subdir}")
    for file in files:
        if file.strip() == "combined_static_feature_extraction.csv":
            file_path = os.path.join(subdir, file)
            print(f"✅ Found CSV: {file_path}")
            try:
                df = pd.read_csv(file_path)

                # Reindex to include all expected columns
                df = df.reindex(columns=ordered_columns, fill_value=0)

                # Add source folder
                df["source_folder"] = os.path.basename(subdir)

                combined_data.append(df)
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e.__class__.__name__} - {e}")

# Combine and export
# Combine and export
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)

    # Drop 'source_folder' column before saving
    if 'source_folder' in final_df.columns:
        final_df = final_df.drop(columns=['source_folder'])

    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Combined CSV saved as: {output_file}")
else:
    print("\n❌ No datasets found. Make sure files are named correctly and exist in subdirectories.")
