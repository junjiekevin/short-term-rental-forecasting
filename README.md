# Airbnb Revenue Models

This repository contains an end-to-end machine learning pipeline and supporting data processing scripts for analyzing, training, and modeling Airbnb rental data.

## Project Structure

- `airbnb_data_analysis__public.py`  
  Cleans and merges multiple Airbnb listing CSVs. Handles missing data, deduplicates listings, and calculates derived metrics like price per guest.

- `airbnb_train_models_public.py`  
  Trains two regression models using Scikit-Learn: one to predict nightly rates and another for cleaning fees. Uses categorical encoding and preprocessing pipelines.

## Features

- Real-world dataset structure (Airbnb listings)
- Feature engineering and seasonal classification
- Scikit-learn Pipelines and ColumnTransformers
- Random Forest Regressors for robust modeling
- Outputs `.pkl` models for downstream simulations

## Getting Started

### Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, joblib

```bash
pip install pandas numpy scikit-learn joblib
```

### Usage

1. Place your Airbnb CSV data in a local `data/` folder.
2. Run `airbnb_data_analysis__public.py` to generate a cleaned combined dataset.
3. Run `airbnb_train_models_public.py` to train and export models to `nightly_rate_model.pkl` and `cleaning_fee_model.pkl`.

## Output Files

- `combined_dataset.csv` – cleaned and merged Airbnb data
- `nightly_rate_model.pkl` – trained model to predict nightly rates
- `cleaning_fee_model.pkl` – trained model to predict cleaning fees

## License

This project is open source and available under the [MIT License](LICENSE).