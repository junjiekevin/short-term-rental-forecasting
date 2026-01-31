
import joblib
import pandas as pd
import logging
from difflib import get_close_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

def load_data_mappings(data_fname="bend_airbnb_combined_cleaned.csv"):
    # Check root for sample data first, then data folder
    sample_path = os.path.join(BASE_DIR, "../sample_airbnb_data.csv")
    data_path = os.path.join(DATA_DIR, data_fname)
    
    final_path = sample_path if os.path.exists(sample_path) else data_path
    
    try:
        df = pd.read_csv(final_path)
        valid_locations = df["Location"].dropna().unique().tolist()
        
        # Calculate mean rating per location as fallback
        if "Location Rating" in df.columns:
            loc_ratings = df.groupby("Location")["Location Rating"].mean().to_dict()
        else:
            loc_ratings = {}
            
        mean_rating = df["Location Rating"].mean() if "Location Rating" in df.columns else 5.0
        return valid_locations, loc_ratings, mean_rating
    except FileNotFoundError:
        logger.warning(f"Data file {data_path} not found. using defaults.")
        return [], {}, 5.0

def get_user_input(prompt, type_=str, default=None):
    while True:
        user_in = input(f"{prompt} " + (f"[{default}]: " if default else ": ")).strip()
        if not user_in and default is not None:
            return default
        try:
            return type_(user_in)
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_.__name__}.")

def normalize_property_type(ptype):
    ptype = ptype.lower()
    if "townhouse" in ptype: return "Townhouse"
    if "condo" in ptype: return "Condo"
    if "apartment" in ptype or "rental unit" in ptype: return "Apartment"
    return "House"

def normalize_location(user_input, valid_locations, default_loc="Old Farm District"):
    if not valid_locations:
        return user_input
    match = get_close_matches(user_input, valid_locations, n=1, cutoff=0.6)
    if match:
        return match[0]
    print(f"Location not recognized. Using default: {default_loc}")
    return default_loc

def predict_with_robustness(model, input_df):
    """
    Ensures input_df has exactly the columns the model expects, filling 0 for missing ones.
    """
    try:
        # Check model features
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
        elif hasattr(model, "steps"): # Pipeline
            # Check the first step (usually preprocessor) or final estimator
             # Try to find the step that tracks features
             # Sklearn pipelines usually expose feature_names_in_ if fit
             expected_features = model.feature_names_in_
        else:
            # Fallback for older sklearn or incompatible models
             expected_features = input_df.columns 
        
        # Align columns
        payload = input_df.copy()
        
        # Add missing columns with 0
        for col in expected_features:
            if col not in payload.columns:
                payload[col] = 0
                
        # Drop extra columns
        payload = payload[[c for c in expected_features if c in payload.columns]]
        
        # Reorder exactly
        payload = payload[expected_features]
        
        return model.predict(payload)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0.0

def run_simulator():
    print("\n--- Airbnb Profit Simulator ---")
    
    # Load assets
    try:
        nightly_model = joblib.load(os.path.join(MODELS_DIR, "nightly_rate_model.pkl"))
        cleaning_model = joblib.load(os.path.join(MODELS_DIR, "cleaning_fee_model.pkl"))
        logger.info("Models loaded.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}. Please run the training script first.")
        return

    valid_locations, loc_ratings_map, global_mean_loc_rating = load_data_mappings()

    # Inputs
    brs = get_user_input("Bedrooms", int, 3)
    baths = get_user_input("Bathrooms", float, 2.0)
    guests = get_user_input("Guests allowed", int, 6)
    superhost = get_user_input("Superhost (1=Yes, 0=No)", int, 0)
    
    loc_input = get_user_input("Location", str)
    location = normalize_location(loc_input, valid_locations)
    
    prop_input = get_user_input("Property Type", str, "House")
    prop_type = normalize_property_type(prop_input)
    
    rating = get_user_input("Current Rating", float, 4.8)
    reviews = get_user_input("Number of Reviews", int, 10)
    
    # Financials
    rent_cost = get_user_input("Monthly Rent/Mortgage", float, 2000.0)
    utilities = get_user_input("Utilites Cost", float, 200.0)
    misc = get_user_input("Misc Cost", float, 100.0)
    
    cleaning_charge_user = get_user_input("Cleaning Fee Charged to Guest", float, 150.0)
    cleaning_labor_cost = get_user_input("Actual Cleaning Labor Cost", float, 120.0)
    
    # Occupancy
    occ_peak = get_user_input("Peak Occupancy (0.0-1.0)", float, 0.85)
    occ_off = get_user_input("Off-Peak Occupancy (0.0-1.0)", float, 0.45)
    
    # Amenities Inputs - Dynamic
    # We should ask for common ones that we know affect price significantly
    # Ideally we'd inspect the model to see which 'Has X' features are important, but standard list is fine
    common_amenities = ["WiFi", "Hot Tub", "Pool", "Air Conditioning", "EV", "Washer", "Dryer", "Parking"]
    amenity_vals = {}
    print("\nSelect Amenities (1=Yes, 0=No):")
    for am in common_amenities:
        amenity_vals[f"Has {am}"] = get_user_input(f"  {am}", int, 0)

    # Prepare DataFrame
    row_base = {
        "No. BRs": brs,
        "No. Beds": brs * 2, # simplified assumption if not asked
        "No. Baths": baths,
        "No. Guests allowed": guests,
        "Superhost Status (1 / 0)": superhost,
        "Rating": rating,
        "No. Reviews": reviews,
        "Property Type Cleaned": prop_type,
        "Location": location,
        "Location Rating": loc_ratings_map.get(location, global_mean_loc_rating),
        # Amenities
        **amenity_vals
    }

    # Seasonal loop
    seasons = ["Peak", "Off-Peak"]
    results = {}
    
    for season in seasons:
        row = row_base.copy()
        row["Season"] = season
        
        # Predict
        input_df = pd.DataFrame([row])
        pred_price = predict_with_robustness(nightly_model, input_df)
        results[season] = pred_price

    # Shoulder (Avg)
    results["Shoulder"] = (results["Peak"] + results["Off-Peak"]) / 2
    
    # Cleaning Fee
    input_df_cf = pd.DataFrame([row_base])
    input_df_cf["Season"] = "Peak" # inputs agnostic to season usually
    pred_cleaning = predict_with_robustness(cleaning_model, input_df_cf)
    
    # Calculate Profit
    print("\n--- Financial Projections ---")
    print(f"Suggested Cleaning Fee: ${pred_cleaning:.2f}")
    
    ann_rev = 0
    ann_cost = 0
    
    occupancies = {"Peak": occ_peak, "Off-Peak": occ_off, "Shoulder": (occ_peak+occ_off)/2}
    
    for season in ["Peak", "Off-Peak", "Shoulder"]:
        nights_per_month = 30 * occupancies[season]
        adr = results[season]
        
        monthly_rev = (nights_per_month * adr) + ((nights_per_month/4) * cleaning_charge_user) # approx 4 night stays
        monthly_exp = rent_cost + utilities + misc + ((nights_per_month/4) * cleaning_labor_cost)
        
        monthly_profit = monthly_rev - monthly_exp
        
        print(f"\n{season} Season:")
        print(f"  ADR: ${adr:.2f}")
        print(f"  Occupancy: {occupancies[season]*100:.0f}%")
        print(f"  Monthly Revenue: ${monthly_rev:,.2f}")
        print(f"  Monthly Profit:  ${monthly_profit:,.2f}")
        
        ann_rev += monthly_rev * 4 # 4 months per season approx
        ann_cost += monthly_exp * 4
        
    print("\n--- Annual Summary ---")
    print(f"Total Revenue: ${ann_rev:,.2f}")
    print(f"Total Profit:  ${(ann_rev - ann_cost):,.2f}")

if __name__ == "__main__":
    run_simulator()
    