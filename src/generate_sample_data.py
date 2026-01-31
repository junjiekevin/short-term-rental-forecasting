import pandas as pd
import numpy as np
import os

def generate_sample_data(output_path):
    np.random.seed(42)
    n_samples = 1000
    
    locations = ["Old Bend, Bend, OR", "Sunriver, OR", "Summit West, Bend, OR", "Larkspur, Bend, OR", 
                 "Tumalo State Park, Bend, OR", "Seventh Mountain, Bend, OR", "Southwest Bend, Bend, OR"]
    prop_types = ["Entire home", "Entire condo", "Entire townhouse", "Entire cabin", "Room"]
    seasons = ["Peak", "Off-Peak", "Shoulder"]
    
    data = []
    for _ in range(n_samples):
        brs = np.random.randint(1, 6)
        beds = brs + np.random.randint(0, 3)
        baths = float(max(1, brs - np.random.randint(0, 2)))
        guests = brs * 2 + np.random.randint(0, 3)
        
        # Base logic for "synthetic" but realistic pricing
        base_rate = 100 + (brs * 50) + (baths * 30)
        loc = np.random.choice(locations)
        season = np.random.choice(seasons)
        
        # Seasonal multipliers
        mult = 1.0 if season == "Peak" else (0.5 if season == "Off-Peak" else 0.75)
        nightly_rate = (base_rate * mult) + np.random.normal(0, 20)
        cleaning_fee = (50 + brs * 40) + np.random.normal(0, 10)
        
        row = {
            "Property name": f"Sample Property {np.random.randint(1000, 9999)}",
            "No. BRs": brs,
            "No. Beds": beds,
            "No. Baths": baths,
            "No. Guests allowed": guests,
            "Nightly Rate": round(max(50, nightly_rate), 2),
            "Cleaning Fee": round(max(20, cleaning_fee), 2),
            "Rating": round(np.random.uniform(3.5, 5.0), 2),
            "No. Reviews": np.random.randint(0, 300),
            "Superhost Status (1 / 0)": np.random.choice([0, 1], p=[0.7, 0.3]),
            "Amenities": "WiFi, Parking, TV, Kitchen, Washer, Dryer",
            "Pets (1 / 0)": np.random.choice([0, 1]),
            "Property Type": np.random.choice(prop_types),
            "Location": loc,
            "Location Rating": round(np.random.uniform(4.0, 5.0), 1),
            "Price_per_Guest": 0, # To be calculated
            "Link": f"https://www.airbnb.com/rooms/sample_{np.random.randint(1, 100000)}",
            "Season": season
        }
        row["Price_per_Guest"] = round(row["Nightly Rate"] / row["No. Guests allowed"], 2)
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} fake data points to {output_path}")

if __name__ == "__main__":
    generate_sample_data("sample_airbnb_data.csv")
