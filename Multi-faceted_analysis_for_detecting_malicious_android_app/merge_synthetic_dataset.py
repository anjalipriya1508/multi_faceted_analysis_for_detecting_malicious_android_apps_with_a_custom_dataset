import pandas as pd

# 📂 Load CSV files
print("📥 Loading benign synthetic csv file: 'begnin_synthetic_data_full.csv' ...")
df1 = pd.read_csv('androZoo_dataset_analysis/dataset_creation/begnin_synthetic_data_full.csv')
print(f"🔢 Rows in benign CSV: {len(df1)}")

print("📥 Loading malicious synthetic csv file: 'malicious_synthetic_data_full.csv' ...")
df2 = pd.read_csv('androZoo_dataset_analysis/dataset_creation/malicious_synthetic_data_full.csv')
print(f"🔢 Rows in malicious CSV: {len(df2)}")

# 🔤 Normalize column names (strip spaces and lowercase)
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# 🔄 Rename 'file_name' to 'app_name'
df1.rename(columns={'file_name': 'app_name'}, inplace=True)
df2.rename(columns={'file_name': 'app_name'}, inplace=True)

# 🏷️ Add label column (temporarily, will reorder later)
df1['label'] = 0  # Benign
df2['label'] = 1  # Malicious

required_col = 'app_name'

# 📊 Get column sets
cols_df1 = set(df1.columns)
cols_df2 = set(df2.columns)

# 🧮 Print number of columns in each file
print(f"📄 Columns in benign CSV: {len(cols_df1)}")
print(f"📄 Columns in malicious CSV: {len(cols_df2)}")

# 🔍 Find common columns
common_columns = cols_df1.intersection(cols_df2)
print(f"✅ Common columns between benign and malicious: {len(common_columns)}")

# 🔁 Union of all columns
all_columns = cols_df1.union(cols_df2)

# ✅ Ensure 'app_name' and 'label' are present
all_columns.update([required_col, 'label'])

# 🧹 Convert to list, place 'app_name' first, 'label' last
all_columns = list(all_columns)
all_columns.remove(required_col)
all_columns.remove('label')
all_columns = [required_col] + sorted(all_columns) + ['label']

# 🔄 Reindex dataframes, fill missing values with 0
df1 = df1.reindex(columns=all_columns, fill_value=0)
df2 = df2.reindex(columns=all_columns, fill_value=0)

# 🧼 Replace 0 in 'app_name' with empty string
df1[required_col] = df1[required_col].replace(0, '')
df2[required_col] = df2[required_col].replace(0, '')

# 🔗 Concatenate dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# 🧾 Show final shape
print(f"📊 Merged dataframe: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

# 💾 Save to CSV
output_path = 'androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv'
merged_df.to_csv(output_path, index=False)

print(f"✅ Merge completed! 🎉 'app_name' is the first column and 'label' is the last column.")
print(f"📁 Output saved to: {output_path}")
