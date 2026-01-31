
# Airbnb Price Optimization & Profit Simulator

Predict Airbnb rental income, optimize cleaning fees, and simulate annual profits using Machine Learning.

---

## 🚀 Overview
This project provides a complete pipeline for Airbnb hosts and investors to:
1. **Analyze** historical listing data.
2. **Train** predictive models for Nightly Rates and Cleaning Fees based on property features (Bedrooms, Baths, Amenities, Location).
3. **Simulate** annual profits across Peak, Off-Peak, and Shoulder seasons.

To ensure immediate usability, this repository includes a **1,000-row synthetic sample dataset** (`sample_airbnb_data.csv`).

## 📁 Project Structure
- `src/`: Core logic and scripts.
  - `airbnb_train_models_public.py`: Trains Random Forest models with hyperparameter tuning.
  - `airbnb_profit_simulator_public.py`: Interactive CLI tool for profit estimation.
  - `airbnb_data_analysis__public.py`: Pre-processing and cleaning pipeline.
  - `generate_sample_data.py`: Utilities for creating synthetic testing data.
- `models/`: (*Gitignored*) Location for trained `.pkl` model files.
- `data/`: (*Gitignored*) Location for private datasets.
- `logs/`: (*Gitignored*) Performance reports and project logs.
- `sample_airbnb_data.csv`: A synthetic dataset for testing the pipeline (Fake data).

## 🛠️ Installation
Ensure you have Python 3.8+ installed, then install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## 📖 Usage Guide

### 1. Data Analysis (Optional)
If you have multiple raw CSVs from scraping or data providers, use this script to merge and clean them:
```bash
python src/airbnb_data_analysis__public.py
```

### 2. Model Training
Train the prediction engines. By default, it will use `sample_airbnb_data.csv` if no parent data is found.
```bash
python src/airbnb_train_models_public.py
```
*The script performs K-Fold Cross-Validation and logs metrics (R², MAE, RMSE) and Feature Importance to `logs/model_performance.txt`.*

### 3. Profit Simulation
Launch the interactive simulator to estimate revenue for a property:
```bash
python src/airbnb_profit_simulator_public.py
```
*You will be prompted for property features (BRs, Baths, Location) and financial overheads (Mortgage, Utilities).*

## 📊 Performance & Methodology
- **Model**: Random Forest Regressor.
- **Validation**: 3-Fold Cross-Validation during training.
- **Feature Engineering**: Automatic parsing of 'Amenities' using Multi-Label Binarization.
- **Data Augmentation**: Includes logic to synthetically generate seasonal pricing variations (Peak/Off-Peak/Shoulder) if only peak data is provided.

## ⚠️ Important Note
The provided `sample_airbnb_data.csv` is **entirely synthetic**. It is intended for code verification and demonstration only. For real-world investment decisions, please use a legally obtained dataset from your local market.

*Note: This project does not support or encourage unauthorized scraping of the Airbnb platform.*