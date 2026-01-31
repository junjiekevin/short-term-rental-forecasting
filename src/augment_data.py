import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def augment_data(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        return

    df_base = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df_base)} rows of Peak data.")

    def add_jitter(series, multiplier, noise_level=0.03):
        """Applies multiplier and adds organic random noise."""
        # multiplier: basic scaling (0.5 for off-peak, 0.75 for shoulder)
        # noise: normal distribution around 1.0 with noise_level standard deviation
        noise = np.random.normal(1.0, noise_level, size=len(series))
        return (series * multiplier * noise).round(2)

    # 1. Peak Data (Original)
    peak = df_base.copy()
    peak["Season"] = "Peak"
    
    # 2. Off-Peak Data (50% of Peak)
    off_peak = df_base.copy()
    off_peak["Season"] = "Off-Peak"
    off_peak["Nightly Rate"] = add_jitter(off_peak["Nightly Rate"], 0.50)
    # Cleaning fees usually stay similar but might drop slightly in off-peak
    off_peak["Cleaning Fee"] = add_jitter(off_peak["Cleaning Fee"], 0.90) 

    # 3. Shoulder Data (Mean of Peak and Off-Peak = 75%)
    shoulder = df_base.copy()
    shoulder["Season"] = "Shoulder"
    shoulder["Nightly Rate"] = add_jitter(shoulder["Nightly Rate"], 0.75)
    shoulder["Cleaning Fee"] = add_jitter(shoulder["Cleaning Fee"], 0.95)

    # Combine
    augmented_df = pd.concat([peak, off_peak, shoulder], ignore_index=True)
    
    # Final cleanup: ensure no negative prices (though unlikely with jitter)
    augmented_df["Nightly Rate"] = augmented_df["Nightly Rate"].clip(lower=20)
    
    augmented_df.to_csv(output_file, index=False)
    logger.info(f"Data augmentation complete. Saved {len(augmented_df)} rows to {output_file}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT = os.path.join(BASE_DIR, "../data/bend_airbnb_combined_cleaned.csv")
    OUTPUT = os.path.join(BASE_DIR, "../data/bend_airbnb_augmented.csv")
    
    augment_data(INPUT, OUTPUT)
