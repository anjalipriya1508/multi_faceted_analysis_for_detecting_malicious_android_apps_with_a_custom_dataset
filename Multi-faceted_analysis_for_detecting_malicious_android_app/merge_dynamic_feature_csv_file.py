import pandas as pd

# 📂 Load CSV files
df1 = pd.read_csv('androZoo_dataset_analysis/begnin_dynamic_feature_extraction.csv')
df2 = pd.read_csv('androZoo_dataset_analysis/malicious_dynamic_features_extraction.csv')

# 🔤 Normalize column names (strip spaces and lowercase)
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# 🔄 Rename 'file_name' to 'app_name'
df1.rename(columns={'file_name': 'app_name'}, inplace=True)
df2.rename(columns={'file_name': 'app_name'}, inplace=True)

required_col = 'app_name'

# 📊 Get column sets
cols_df1 = set(df1.columns)
cols_df2 = set(df2.columns)

# 🧮 Print number of columns in each file
print(f"📄 Columns in 'begnin_dynamic_feature_extraction.csv': {len(cols_df1)}")
print(f"📄 Columns in 'malicious_dynamic_features_extraction.csv': {len(cols_df2)}")

# 🔍 Find common columns
common_columns = cols_df1.intersection(cols_df2)
print(f"✅ Common columns between benign and malicious: {len(common_columns)}")

# 🔁 Union of all columns
all_columns = cols_df1.union(cols_df2)

# ✅ Ensure 'app_name' is present
if required_col not in all_columns:
    all_columns.add(required_col)

# 🧹 Convert to list and sort 'app_name' to be first
all_columns = list(all_columns)
if required_col in all_columns:
    all_columns.remove(required_col)
all_columns = [required_col] + all_columns

# 🔄 Reindex dataframes, fill missing values with 0
df1 = df1.reindex(columns=all_columns, fill_value=0)
df2 = df2.reindex(columns=all_columns, fill_value=0)

# 🧼 Replace 0 in 'app_name' with empty string
df1[required_col] = df1[required_col].replace(0, '')
df2[required_col] = df2[required_col].replace(0, '')

# 🔗 Concatenate dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# 🧮 Show final column count
print(f"🧾 Total columns in merged dataframe: {merged_df.shape[1]}")

# 💾 Save to CSV
merged_df.to_csv('androZoo_dataset_analysis/combined_dynamic_feature_extraction.csv', index=False)

print("✅ Merge completed! 🎉 'app_name' is the first column in 'combined_dynamic_feature_extraction.csv'. 📁")
