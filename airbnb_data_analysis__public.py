import pandas as pd
import glob
import os

# Folder where your CSVs are stored
folder_path = # Include folder path here
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Load and combine all CSVs
dfs = []
for file in csv_files:
    df = # Load dataset (place in 'data/' directory or update path)
    pd.read_csv(file)
    df['Source File'] = os.path.basename(file)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# Ensure numeric fields are correct
all_data["Nightly Rate"] = pd.to_numeric(all_data["Nightly Rate"], errors="coerce")
all_data["No. Guests allowed"] = pd.to_numeric(all_data["No. Guests allowed"], errors="coerce")

# Drop duplicates by Link, keeping the one with the highest nightly rate
before = len(all_data)
all_data.sort_values(by="Nightly Rate", ascending=False, inplace=True)
all_data.drop_duplicates(subset='Link', keep='first', inplace=True)
after = len(all_data)

# If Price_per_Guest exists and has NaNs, recalculate them
if "Price_per_Guest" in all_data.columns:
    missing_ppg = all_data["Price_per_Guest"].isna()
    all_data.loc[missing_ppg, "Price_per_Guest"] = all_data.loc[missing_ppg].apply(
        lambda row: round(row["Nightly Rate"] / row["No. Guests allowed"], 2)
        if pd.notnull(row["Nightly Rate"]) and pd.notnull(row["No. Guests allowed"]) and row["No. Guests allowed"] > 0
        else None,
        axis=1
    )
else:
    # If column missing entirely, create it
    all_data["Price_per_Guest"] = all_data.apply(
        lambda row: round(row["Nightly Rate"] / row["No. Guests allowed"], 2)
        if pd.notnull(row["Nightly Rate"]) and pd.notnull(row["No. Guests allowed"]) and row["No. Guests allowed"] > 0
        else None,
        axis=1
    )

# Save to CSV
output_path = os.path.join(folder_path, 'combined_dataset.csv') # Change name when necessary
all_data.to_csv(output_path, index=False)

# Summary
print(f"Files merged: {len(csv_files)}")
print(f"Total listings before deduplication: {before}")
print(f"Unique listings after deduplication: {after}")
print("'Price_per_Guest' column verified and cleaned.")
print(f"Cleaned dataset saved to: {output_path}")