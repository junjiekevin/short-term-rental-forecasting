
import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(folder_path):
    """
    Loads all CSV files from the specified folder.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files found in {folder_path}")
        return pd.DataFrame()

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['Source File'] = os.path.basename(file)
            dfs.append(df)
            logger.info(f"Loaded {file} with {len(df)} rows.")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")

    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def clean_data(df):
    """
    Cleans the dataframe: handles numeric conversions, deduplication, and derived metrics.
    """
    if df.empty:
        logger.warning("Empty dataframe provided for cleaning.")
        return df

    before_count = len(df)
    
    # Ensure numeric fields
    for col in ["Nightly Rate", "No. Guests allowed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates by Link, keeping the one with the highest nightly rate
    # (Assuming highest rate might be the most complete or peak pricing data, 
    # though 'latest' might be better if we had timestamps. Sticking to user logic.)
    if "Link" in df.columns and "Nightly Rate" in df.columns:
        df.sort_values(by="Nightly Rate", ascending=False, inplace=True)
        df.drop_duplicates(subset='Link', keep='first', inplace=True)
    
    # Calculate Price per Guest
    if "Nightly Rate" in df.columns and "No. Guests allowed" in df.columns:
        # Vectorized calculation is faster than apply
        df["Price_per_Guest"] = df["Nightly Rate"] / df["No. Guests allowed"]
        df["Price_per_Guest"] = df["Price_per_Guest"].round(2)
        
    logger.info(f"Data cleaned. Rows before: {before_count}, after: {len(df)}")
    return df

def save_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {output_path}: {e}")

if __name__ == "__main__":
    # improved default path handling
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER_PATH = os.path.join(BASE_DIR, "../data") 
    OUTPUT_FILE = "combined_dataset.csv"
    
    logger.info("Starting data analysis pipeline...")
    
    all_data = load_data(FOLDER_PATH)
    if not all_data.empty:
        cleaned_data = clean_data(all_data)
        save_data(cleaned_data, os.path.join(FOLDER_PATH, OUTPUT_FILE))
    else:
        logger.error("No data to process.")
        