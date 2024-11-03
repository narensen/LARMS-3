import pandas as pd
import os

# Specify the directory containing your CSV files
directory = '/home/naren/Documents/lifey.llm/corpus'  # Replace with your directory path

# Create an empty list to store individual dataframes
dfs = []

# Read all CSV files from the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all dataframes vertically
merged_df = pd.concat(dfs, ignore_index=True)

# Remove any duplicate rows if needed
merged_df = merged_df.drop_duplicates()

# Save the merged dataset
merged_df.to_csv('/home/naren/Documents/lifey.llm/corpus/merged_dataset.csv', index=False)

# Print information about the merge
print(f"Number of files merged: {len(dfs)}")
print(f"Final dataset shape: {merged_df.shape}")

# Optional: Display first few rows of merged dataset
print("\nFirst few rows of merged dataset:")
print(merged_df.head())