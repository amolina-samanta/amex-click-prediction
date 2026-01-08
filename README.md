# Amex Click Prediction Project

A machine learning project for predicting user clicks on offers using LightGBM, XGBoost, and CatBoost models with ensemble learning techniques.

##  Project Overview

This project implements a binary classification model to predict whether users will click on specific offers. The solution uses an ensemble approach combining multiple gradient boosting models with dimensionality reduction techniques (PCA) and stacking for improved performance.

##  Project Structure

```
amex-click-prediction/
│
├── Amex2.ipynb                    # Data preprocessing and feature engineering
├── Amex_lgbm_w_PCA.ipynb         # LightGBM model with PCA
└── Amex_New.ipynb                 # Ensemble model training (XGBoost, LightGBM, CatBoost)
│
├── data/                              # Data directory (not included in repo)
│   ├── train_data.parquet
│   ├── test_data.parquet
│   ├── add_event.parquet
│   ├── add_trans.parquet
│   └── offer_metadata.parquet
│
├── requirements.txt
├── data_dictionary.csv
├── .gitignore
└── README.md
```

##  Features

- **Data Engineering**: Combines multiple data sources (events, transactions, offer metadata)
- **Feature Engineering**: 
  - Numerical feature scaling and PCA dimensionality reduction
  - Categorical feature encoding
  - Time-based features extraction
- **Model Training**:
  - LightGBM with PCA (95% variance retention)
  - XGBoost, LightGBM, and CatBoost ensemble
  - 5-fold stratified cross-validation
- **Ensemble Methods**:
  - Weighted soft voting
  - Rank-based blending
  - Stacking with Logistic Regression meta-learner

##  Model Performance

- **LightGBM with PCA**: Validation AUC ~0.926
- **MAP@7 Score**: ~0.629 (Mean Average Precision at 7)
- Individual model performances tracked through out-of-fold predictions

##  Installation

### Prerequisites
- Python 3.8+
- Google Colab (for original notebooks) or local Jupyter environment

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amex-click-prediction.git
cd amex-click-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up data directories:
```bash
mkdir -p data models oof_preds submissions
```

## 📦 Dependencies

- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- joblib

##  Usage

### Data Preprocessing

Run `Amex2.ipynb` to:
- Merge multiple data sources
- Create combined train and test datasets
- Apply feature engineering

### Model Training

**Option 1: LightGBM with PCA**
```python
# Run Amex_lgbm_w_PCA.ipynb
# This notebook includes:
# - PCA dimensionality reduction
# - LightGBM training with categorical features
# - MAP@7 evaluation
```

**Option 2: Full Ensemble**
```python
# Run Amex_New.ipynb
# This notebook includes:
# - 5-fold cross-validation
# - Multiple model training (XGB, LGB, CAT)
# - Meta-learner stacking
# - Final ensemble predictions
```

##  Evaluation Metrics

- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve
- **MAP@7**: Mean Average Precision at 7 (ranking metric)

##  Key Configuration

### Model Parameters

**LightGBM:**
```python
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 64,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
```

**XGBoost:**
```python
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

##  Notes

- Original notebooks were designed for Google Colab with Google Drive integration
- Data files are stored in Parquet format for efficient storage and loading
- Sampling strategies used to handle class imbalance (25% negative sampling)
- PCA retains 95% of variance while reducing dimensionality

##  Contributing

This was a hackathon project. Feel free to fork and experiment with different approaches!

##  License

This project is open source and available under the MIT License.

##  Important

- Data files are not included in the repository due to size and privacy considerations
- Ensure you have sufficient computational resources for training (models trained with ~220K samples)
- Original data paths reference Google Drive - update paths for local execution

##  Acknowledgments

- Hackathon organizers
- Libraries: scikit-learn, LightGBM, XGBoost, CatBoost
