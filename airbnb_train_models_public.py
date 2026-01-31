# Model Training
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

# Load cleaned dataset
df = # Load dataset (place in 'data/' directory or update path)
pd.read_csv("{include file name or file path here}")

# Add season column if missing — ensure values: Peak, Off-Peak, Shoulder
if "Season" not in df.columns:
    raise ValueError("Missing 'Season' column in the dataset. Please tag each listing by season first.")

# 'Property Type Cleaned'
df["Property Type Cleaned"] = df["Property Type"].str.lower().map({
    "entire home": "House",
    "entire condo": "Condo",
    "entire townhouse": "Townhouse",
    "entire rental unit": "Apartment",
    "entire guest suite": "House",
    "entire cabin": "House",
    "entire guesthouse": "House",
    "entire bungalow": "House",
    "entire cottage": "House",
    "entire vacation home": "House",
    "room": "Room",
    "entire chalet": "Other",
    "tiny home": "Other",
    "camper/rv": "Other",
    "dome": "Other"
}).fillna("House")

# Drop incomplete records (essential features only)
df_model = df.dropna(subset=[
    "No. BRs", "No. Beds", "No. Baths", "No. Guests allowed",
    "Nightly Rate", "Cleaning Fee", "Rating", "No. Reviews",
    "Superhost Status (1 / 0)", "Has WiFi", "Has Washer", "Has Dryer", "Has Parking",
    "Has AC", "Has EV Charging", "Has Pool", "Has Hot Tub", 
    "Pets (1 / 0)", "Property Type Cleaned", "Location", "Location Rating", "Season"
])

# Features to include
features = [
    "No. BRs", "No. Beds", "No. Baths", "No. Guests allowed",
    "Nightly Rate", "Cleaning Fee", "Rating", "No. Reviews",
    "Superhost Status (1 / 0)", "Has WiFi", "Has Washer", "Has Dryer", "Has Parking",
    "Has AC", "Has EV Charging", "Has Pool", "Has Hot Tub", 
    "Pets (1 / 0)", "Property Type Cleaned", "Location", "Location Rating", "Season"
]

# Target variables
X = df_model[features]
y_nightly = df_model["Nightly Rate"]
y_cleaning = df_model["Cleaning Fee"]

# Categorical preprocessing
categorical = ["Location", "Property Type Cleaned", "Season"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Model pipelines
pipeline_nightly = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline_cleaning = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train_n, y_test_n = train_test_split(X, y_nightly, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(X, y_cleaning, test_size=0.2, random_state=42)

# Train models
pipeline_nightly.fit(X_train, y_train_n)
pipeline_cleaning.fit(X_train, y_train_c)

# Save trained models
joblib.dump(pipeline_nightly, "nightly_rate_model.pkl")
joblib.dump(pipeline_cleaning, "cleaning_fee_model.pkl")

# Evaluate
y_train_pred = pipeline_nightly.predict(X_train)
y_test_pred = pipeline_nightly.predict(X_test)

print("Models trained and saved. Ready for simulator.")
print(f"Train R-squared: {r2_score(y_train_n, y_train_pred):.4f}")
print(f"Test R-squared: {r2_score(y_test_n, y_test_pred):.4f}")