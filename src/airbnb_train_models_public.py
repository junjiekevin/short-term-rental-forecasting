
import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import re
from datetime import datetime

def load_and_prep_data(filepath):
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error("File not found.")
        return None

    # Ensure Season is present (it should be in augmented data)
    if "Season" not in df.columns:
        logger.warning("'Season' column missing from dataset. Check augmentation script.")
        df["Season"] = "Peak"
    
    # Analyze Season Distribution
    season_counts = df["Season"].value_counts()
    logger.info(f"Season Distribution:\n{season_counts}")
    
    # Clean Property Type
    df["Property Type Cleaned"] = df["Property Type"].astype(str).str.lower().map({
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
    
    return df

def parse_amenities(df):
    """
    Parses 'All Amenities' column containing comma-separated strings into binary columns.
    Returns: df with new columns, list of new amenity column names
    """
    if "All Amenities" not in df.columns:
        return df, []

    # Fill NaNs
    df["All Amenities"] = df["All Amenities"].fillna("")
    
    # Split strings
    amenities_list = df["All Amenities"].apply(lambda x: [item.strip() for item in x.split(',')] if x else [])
    
    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(amenities_list)
    amenity_cols = [f"Has {cls}" for cls in mlb.classes_]
    
    amenities_df = pd.DataFrame(amenities_encoded, columns=amenity_cols, index=df.index)
    
    # Drop original and join
    df = df.join(amenities_df)
    
    logger.info(f"Parsed {len(amenity_cols)} amenity features.")
    return df, amenity_cols

def train_and_evaluate(X, y, model_name="model"):
    # Preprocessing
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy='median'), numeric_cols),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])
    
    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter Tuning
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10]
    }
    
    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training {model_name}...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    logger.info(f"--- {model_name} Results ---")
    logger.info(f"Best Params: {search.best_params_}")
    logger.info(f"Best CV Score (Validation R2): {search.best_score_:.4f}")  # This is the "Validation" score
    logger.info(f"Test R2: {r2:.4f}")
    logger.info(f"Test MAE: {mae:.2f}")
    logger.info(f"Test RMSE: {rmse:.2f}")

    # Feature Importance
    feature_importance_dict = {}
    try:
        # Access steps
        model_step = best_model.named_steps['model']
        preprocessor_step = best_model.named_steps['preprocessor']
        
        # Get feature names
        feature_names = preprocessor_step.get_feature_names_out()
        
        # Get importances
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            feature_importance_dict = dict(zip(feature_names, importances))
            # Sort
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    metrics = {
        "Model": model_name,
        "Model Type": best_model.named_steps['model'].__class__.__name__,
        "Num Samples": len(X),
        "Validation R2 (CV)": search.best_score_,
        "Test R2": r2,
        "Test MAE": mae,
        "Test MSE": mse,
        "Test RMSE": rmse,
        "Best Params": search.best_params_,
        "Feature Importance": feature_importance_dict
    }
    
    return best_model, metrics

if __name__ == "__main__":
    # Relative paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")
    MODELS_DIR = os.path.join(BASE_DIR, "../models")
    LOGS_DIR = os.path.join(BASE_DIR, "../logs")
    
    # Ensure dirs exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Priority: 1. Sample data in root, 2. Augmented data in data/, 3. Fallback
    SAMPLE_DATA = os.path.join(BASE_DIR, "../sample_airbnb_data.csv")
    AUGMENTED_DATA = os.path.join(DATA_DIR, "bend_airbnb_augmented.csv")
    
    if os.path.exists(SAMPLE_DATA):
        DATA_FILE = SAMPLE_DATA
    elif os.path.exists(AUGMENTED_DATA):
        DATA_FILE = AUGMENTED_DATA
    else:
        DATA_FILE = os.path.join(DATA_DIR, "bend_airbnb_combined_cleaned.csv")
    
    df = load_and_prep_data(DATA_FILE)
    
    if df is not None:
        # Parse amenities
        df, amenity_cols = parse_amenities(df)
        
        # Base Features
        base_features = [
            "No. BRs", "No. Beds", "No. Baths", "No. Guests allowed",
            "Superhost Status (1 / 0)", "Rating", "No. Reviews",
            "Property Type Cleaned", "Location", "Location Rating", "Season"
        ]
        
        # Add parsed amenities to feature list
        available_amenity_cols = [c for c in amenity_cols if c in df.columns]
        features = base_features + available_amenity_cols
        
        # Drop rows missing targets or essential features
        df = df.dropna(subset=["Nightly Rate", "Cleaning Fee"] + [f for f in base_features if f in df.columns])
        
        X = df[features]
        y_nightly = df["Nightly Rate"]
        y_cleaning = df["Cleaning Fee"]
        
        all_metrics = []

        # Train Nightly Rate Model
        model_nightly, metrics_nightly = train_and_evaluate(X, y_nightly, "Nightly Rate Model")
        joblib.dump(model_nightly, os.path.join(MODELS_DIR, "nightly_rate_model.pkl"))
        all_metrics.append(metrics_nightly)
        
        # Train Cleaning Fee Model
        model_cleaning, metrics_cleaning = train_and_evaluate(X, y_cleaning, "Cleaning Fee Model")
        joblib.dump(model_cleaning, os.path.join(MODELS_DIR, "cleaning_fee_model.pkl"))
        all_metrics.append(metrics_cleaning)
        
        # Save Metrics Report
        report_path = os.path.join(LOGS_DIR, "model_performance.txt")
        with open(report_path, "w") as f:
            f.write("--- Model Performance Report ---\n")
            f.write(f"Data Source: {DATA_FILE}\n")
            f.write("-" * 30 + "\n")
            for m in all_metrics:
                f.write(f"\nModel: {m['Model']}\n")
                f.write(f"Model Type: {m['Model Type']}\n")
                f.write(f"Num Samples: {m['Num Samples']}\n")
                f.write(f"Validation R2 (CV): {m['Validation R2 (CV)']:.4f}\n")
                f.write(f"Test R2 Score: {m['Test R2']:.4f}\n")
                f.write(f"Test MAE: {m['Test MAE']:.4f}\n")
                f.write(f"Test RMSE: {m['Test RMSE']:.4f} (Lower is better)\n")
                f.write(f"Best Params: {m['Best Params']}\n")
                
                if m["Feature Importance"]:
                    f.write("\nTop 10 Influential Features:\n")
                    count = 0
                    for feat, imp in m["Feature Importance"].items():
                        if count >= 10: break
                        f.write(f"  - {feat}: {imp:.4f}\n")
                        count += 1
                f.write("-" * 30 + "\n")
        
        logger.info(f"Models saved to {MODELS_DIR}")
        logger.info(f"Performance report saved to {report_path}")
        