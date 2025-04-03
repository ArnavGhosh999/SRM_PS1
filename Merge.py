import pandas as pd
import os
import glob
from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).parent

# Find all CSV files in the directory
csv_files = glob.glob(str(current_dir / "*.csv"))

# Initialize an empty list to store dataframes
dfs = []

# Process each CSV file
for file_path in csv_files:
    try:
        # Get the zone name from the filename (without extension)
        zone_name = Path(file_path).stem
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add zone column
        df['Zone'] = zone_name
        
        # Append to list
        dfs.append(df)
        print(f"Successfully loaded {zone_name}")
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

if not dfs:
    print("No CSV files were successfully loaded!")
    exit()

# Merge all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Handle missing values
# Fill numeric columns with median
numeric_columns = merged_df.select_dtypes(include=['int64', 'float64']).columns
merged_df[numeric_columns] = merged_df[numeric_columns].fillna(merged_df[numeric_columns].median())

# Fill categorical columns with mode
categorical_columns = merged_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0])

# Save the merged dataframe
output_path = current_dir / "merged_data.csv"
merged_df.to_csv(output_path, index=False)

# Display basic information about the merged dataset
print("\nMerged Dataset Information:")
print(f"Total rows: {len(merged_df)}")
print(f"Total columns: {len(merged_df.columns)}")
print("\nColumns and their data types:")
print(merged_df.dtypes)
print("\nFirst few rows of the merged dataset:")
print(merged_df.head())
print(f"\nMerged data saved to: {output_path}")
