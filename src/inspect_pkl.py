
import joblib
import pandas as pd
import numpy as np
import os

def inspect_pkl(filename):
    print(f"--- Inspecting {filename} ---")
    if not os.path.exists(filename):
        print("File not found.")
        return

    try:
        model = joblib.load(filename)
        print(f"Type: {type(model)}")
        
        # Check if it's a pipeline
        if hasattr(model, 'steps'):
            print("Is a Pipeline.")
            for name, step in model.steps:
                print(f"Step: {name} -> {type(step)}")
                if hasattr(step, 'feature_names_in_'):
                    print(f"  Feature Names In: {list(step.feature_names_in_)}")
                if hasattr(step, 'transformers_'): # ColumnTransformer
                    print("  Transformers:")
                    for t_name, t_trans, t_cols in step.transformers_:
                        print(f"    {t_name}: {t_trans} on {t_cols}")
        
        # Check for feature names if available directly
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature Names In: {list(model.feature_names_in_)}")

    except Exception as e:
        print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    inspect_pkl("nightly_rate_model.pkl")
    inspect_pkl("cleaning_fee_model.pkl")
