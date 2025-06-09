import pandas as pd

# Load CSV files
df1 = pd.read_csv('androZoo_dataset_analysis/begnin_static_feature_extraction.csv')
df2 = pd.read_csv('androZoo_dataset_analysis/malicious_static_feature_extraction.csv')
# Add more files if needed, e.g. df3 = pd.read_csv('path_to_file3.csv')

# Normalize column names
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()
# df3.columns = df3.columns.str.strip().str.lower()  # if used

required_col = 'app_name'

# Check and remove duplicate columns before processing
df1 = df1.loc[:, ~df1.columns.duplicated()]
df2 = df2.loc[:, ~df2.columns.duplicated()]
# df3 = df3.loc[:, ~df3.columns.duplicated()]  # if used

# Print number of columns
print(f"Number of columns in begnin_static_feature_extraction.csv: {df1.shape[1]}")
print(f"Number of columns in malicious_static_feature_extraction.csv: {df2.shape[1]}")

# Find common columns
common_columns = set(df1.columns).intersection(set(df2.columns))
print(f"Number of common columns between file1 and file2: {len(common_columns)}")

# Union of all columns
all_columns = set(df1.columns).union(set(df2.columns))
# all_columns = all_columns.union(df3.columns)  # if used

# Ensure 'app_name' is first
all_columns = list(all_columns)
if required_col in all_columns:
    all_columns.remove(required_col)
all_columns = [required_col] + sorted(all_columns)  # sorted for consistency

# Reindex with fill_value=0
df1 = df1.reindex(columns=all_columns, fill_value=0)
df2 = df2.reindex(columns=all_columns, fill_value=0)
# df3 = df3.reindex(columns=all_columns, fill_value=0)  # if used

# Clean app_name column
df1[required_col] = df1[required_col].replace(0, '')
df2[required_col] = df2[required_col].replace(0, '')
# df3[required_col] = df3[required_col].replace(0, '')  # if used

# Merge row-wise
merged_df = pd.concat([df1, df2], ignore_index=True)
# merged_df = pd.concat([df1, df2, df3], ignore_index=True)  # if used

# Output result
print(f"Number of columns in merged dataframe: {merged_df.shape[1]}")

# Save to CSV
merged_df.to_csv('androZoo_dataset_analysis/combined_static_feature_extraction.csv', index=False)
print("Merge completed. 'app_name' is the first column in 'combined_static_feature_extraction.csv'.")
