
# merge_data.py - Dataset Merger for LARMS

This script combines multiple CSV files containing mental health conversation data into a single merged dataset for use with the LARMS (Large Language Models for Remedying Mental Status) system.

## Purpose

The script performs the following operations:
- Reads all CSV files from a specified directory
- Combines them into a single dataset
- Removes any duplicate entries
- Saves the merged result as `merged_dataset.csv`

## Requirements

- Python 3.x
- pandas library (`pip install pandas`)

## File Structure

Your directory should look like this:
```
corpus/
├── merge_data.py
├── dataset1.csv
├── dataset2.csv
├── dataset3.csv
└── ...
```

## Input Data Requirements

Each CSV file should contain:
- A 'Context' column (user messages/queries)
- A 'Response' column (corresponding responses)
- All CSV files should have the same column structure

## Usage

1. Place all your source CSV files in the `corpus` directory

2. Update the directory path in the script:
```python
directory = '/path/to/your/corpus'  # Replace with your actual path
```

3. Run the script:
```bash
python merge_data.py
```

## Output

The script will:
1. Create `merged_dataset.csv` in the specified directory
2. Print a summary including:
   - Number of files merged
   - Final dataset dimensions
   - Preview of the merged data

## Example Output

```
Number of files merged: 3
Final dataset shape: (1000, 2)

First few rows of merged dataset:
                            Context                                           Response
0  I'm feeling really anxious today  I understand that anxiety can be overwhelming...
1  Can't sleep at night              Let's talk about what might be keeping you...
...
```

## Error Handling

- The script will only process files with `.csv` extension
- Ensure all CSV files have consistent column names
- Make sure you have write permissions in the output directory

## Integration with LARMS

This script must be run before starting the main LARMS application to ensure the required `merged_dataset.csv` file is available for the chatbot to function.

## Notes

- Duplicate rows are automatically removed to prevent redundancy
- The script preserves the original column structure
- No index column is included in the output file
