# config.py
"""
Configuration module for fraud detection project.
Centralized settings for paths, model hyperparameters, and feature definitions.
"""

import os

# ============== PROJECT PATHS ==============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Model file paths
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.pkl")

# Data file paths
TRAIN_TRANSACTION_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY_PATH = os.path.join(DATA_DIR, "train_identity.csv")
TEST_TRANSACTION_PATH = os.path.join(DATA_DIR, "test_transaction.csv")

# ============== MODEL HYPERPARAMETERS ==============
# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'is_unbalance': False,  # We'll use scale_pos_weight instead for better control
}

# SMOTE and class balancing
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.5  # Balance ratio for minority class
SCALE_POS_WEIGHT = 15  # For imbalanced datasets; adjust based on fraud ratio

# Training parameters
NUM_BOOSTING_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

# ============== FEATURE DEFINITIONS ==============
# User-facing features for the Streamlit dashboard
USER_FACING_FEATURES = {
    'TransactionAmt': {
        'label': 'Transaction Amount ($)',
        'type': 'number',
        'min': 0.0,         # تم التعديل إلى float ليتوافق مع step
        'max': 100000.0,    # تم التعديل إلى float
        'default': 100.0,   # تم التعديل إلى float
        'step': 0.01,
        'description': 'Amount in USD'
    },
    'card4': {
        'label': 'Card Type',
        'type': 'categorical',
        'options': ['visa', 'mastercard', 'american express', 'discover'],
        'default': 'visa',
        'description': 'Payment-card network'
    },
    'card6': {
        'label': 'Card Category',
        'type': 'categorical',
        'options': ['debit', 'credit', 'debit or credit'],
        'default': 'credit',
        'description': 'Payment-card category'
    },
    'DeviceType': {
        'label': 'Device Type',
        'type': 'categorical',
        'options': ['desktop', 'mobile', 'unknown'],
        'default': 'desktop',
        'description': 'Device used for transaction'
    },
    'P_emaildomain': {
        'label': 'Email Domain',
        'type': 'categorical',
        'options': ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com', 'other'],
        'default': 'gmail.com',
        'description': 'Email domain of purchaser'
    },
    'ProductCD': {
        'label': 'Product Type',
        'type': 'categorical',
        'options': ['W', 'H', 'S', 'C', 'R'],
        'default': 'W',
        'description': 'Anonymized product category code'
    },
'addr1': {
    'label': 'Billing Address Code',
    'type': 'number',
    'min': 0.0,
    'max': 540.0,
    'default': 299.0,
    'step': 1.0,
    'description': 'Anonymized billing-region code'
},
    'hour_of_day': {
        'label': 'Hour of Day (0-23)',
        'type': 'number',
        'min': 0,
        'max': 23,
        'default': 12,
        'step': 1,
        'description': 'Hour when transaction occurred'
    },
    'day_of_week': {
        'label': 'Day of Week',
        'type': 'categorical',
        'options': list(range(7)),
        'default': 0,
        'description': '0 = Monday; 6 = Sunday'
    },
    'is_night_transaction': {
        'label': 'Night Transaction (10PM-6AM)',
        'type': 'categorical',
        'options': [0, 1],
        'default': 0,
        'description': '1 if transaction occurred at night, 0 otherwise'
    },
}

# Features to be auto-filled with defaults/medians (hidden from user)
AUTO_FILLED_FEATURES = {
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10',
}

# ============== UI CONFIGURATION ==============
APP_TITLE = "Fraud Detection Engine"
APP_DESCRIPTION = "Real-time credit card transaction fraud detection powered by advanced Machine Learning"
APP_ICON = "🔐"
PREDICTION_THRESHOLD = 0.5
COLOR_LEGITIMATE = "#2ecc71"  # Green
COLOR_FRAUD = "#e74c3c"       # Red
COLOR_NEUTRAL = "#3498db"     # Blue
