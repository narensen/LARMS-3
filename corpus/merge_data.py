import pandas as pd
import os


directory = '/home/naren/LARMS-1.2/corpus/'
dfs = []
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)


merged_df = pd.concat(dfs, ignore_index=True)


merged_df = merged_df.drop_duplicates()


merged_df.to_csv('/home/naren/LARMS-1.2/corpus/merged_dataset.csv', index=False)

print(f"Number of files merged: {len(dfs)}")
print(f"Final dataset shape: {merged_df.shape}")


print("\nFirst few rows of merged dataset:")
print(merged_df.head())