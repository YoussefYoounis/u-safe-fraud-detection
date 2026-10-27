# train.py

"""

Model training module.

Trains LightGBM classifier with SMOTE for handling class imbalance.

"""



import pandas as pd

import numpy as np

import pickle

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    roc_auc_score, roc_curve, confusion_matrix,

    classification_report, f1_score

)

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import seaborn as sns

from .config import (

    LIGHTGBM_PARAMS, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,

    USE_SMOTE, SMOTE_SAMPLING_STRATEGY, SCALE_POS_WEIGHT,

    NUM_BOOSTING_ROUNDS, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL,

    MODEL_PATH, FEATURE_IMPORTANCE_PATH, REPORTS_DIR

)

from .data_loader import DataLoader

from .preprocessing import DataPreprocessor





class ModelTrainer:

    """Handles model training and evaluation."""



    def __init__(self):

        self.model = None

        self.feature_importance = None

        self.X_train = None

        self.X_val = None

        self.y_train = None

        self.y_val = None

        self.preprocessing_pipeline = None



    def load_and_preprocess_data(self):

        """Load data and apply preprocessing."""

        print("\n" + "="*60)

        print("LOADING AND PREPROCESSING DATA")

        print("="*60)



        # Load data

        loader = DataLoader()

        df = loader.get_merged_train_data()



        # Preprocess

        preprocessor = DataPreprocessor()

        df_processed = preprocessor.preprocess(df, fit=True, target_col='isFraud')



        self.preprocessing_pipeline = preprocessor



        return df_processed



    def split_data(self, df):

        """Split data into train, validation, and test sets."""

        print("\n" + "="*60)

        print("SPLITTING DATA")

        print("="*60)



        # Separate features and target

        X = df.drop(columns=['isFraud'])

        y = df['isFraud']



        print(f"Class distribution:\n{y.value_counts()}")

        print(f"Fraud ratio: {y.mean():.4f}")



        # Train-test split

        X_train, X_test, y_train, y_test = train_test_split(

            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y

        )



        # Further split train into train and validation

        X_train, X_val, y_train, y_val = train_test_split(

            X_train, y_train, test_size=VALIDATION_SIZE,

            random_state=RANDOM_STATE, stratify=y_train

        )



        print(f"\nTrain set size: {X_train.shape}")

        print(f"Validation set size: {X_val.shape}")

        print(f"Test set size: {X_test.shape}")



        self.X_train = X_train

        self.X_val = X_val

        self.y_train = y_train

        self.y_val = y_val



        return X_train, X_val, X_test, y_train, y_val, y_test



    def apply_smote(self):

        """Apply SMOTE to balance training data."""

        if not USE_SMOTE:

            print("SMOTE disabled in config.")

            return self.X_train, self.y_train



        print("\n" + "="*60)

        print("APPLYING SMOTE")

        print("="*60)



        smote = SMOTE(

            sampling_strategy=SMOTE_SAMPLING_STRATEGY,

            random_state=RANDOM_STATE,

        )



        X_train_resampled, y_train_resampled = smote.fit_resample(

            self.X_train, self.y_train

        )



        print(f"Original training set size: {self.X_train.shape}")

        print(f"Resampled training set size: {X_train_resampled.shape}")

        print(f"New class distribution:\n{pd.Series(y_train_resampled).value_counts()}")



        return X_train_resampled, y_train_resampled



    def train_model(self, X_train, y_train):

        """Train LightGBM model."""

        print("\n" + "="*60)

        print("TRAINING LIGHTGBM MODEL")

        print("="*60)



        # Create LightGBM datasets

        train_data = lgb.Dataset(

            X_train, label=y_train,

            feature_name=X_train.columns.tolist()

        )



        val_data = lgb.Dataset(

            self.X_val, label=self.y_val,

            reference=train_data

        )



        # Add scale_pos_weight to params for imbalance handling

        params = LIGHTGBM_PARAMS.copy()

        params['scale_pos_weight'] = SCALE_POS_WEIGHT



        # Train model

        self.model = lgb.train(

            params,

            train_data,

            num_boost_round=NUM_BOOSTING_ROUNDS,

            valid_sets=[train_data, val_data],

            valid_names=['train', 'valid'],

            callbacks=[

                lgb.log_evaluation(period=VERBOSE_EVAL),

                lgb.early_stopping(EARLY_STOPPING_ROUNDS)

            ]

        )



        print(f"\nModel trained with {self.model.num_trees()} trees.")



        return self.model



    def evaluate_model(self, X_test, y_test):

        """Evaluate model performance."""

        print("\n" + "="*60)

        print("EVALUATING MODEL")

        print("="*60)



        # Predictions

        y_pred_proba = self.model.predict(X_test)

        y_pred = (y_pred_proba >= 0.5).astype(int)



        # Metrics

        auc_score = roc_auc_score(y_test, y_pred_proba)

        f1 = f1_score(y_test, y_pred)



        print(f"\nROC-AUC Score: {auc_score:.4f}")

        print(f"F1 Score: {f1:.4f}")



        # Confusion Matrix

        cm = confusion_matrix(y_test, y_pred)

        print(f"\nConfusion Matrix:\n{cm}")



        # Classification Report

        print("\nClassification Report:")

        print(classification_report(y_test, y_pred,

                                   target_names=['Legitimate', 'Fraud']))



        # ROC Curve

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)



        plt.figure(figsize=(10, 6))

        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('ROC Curve')

        plt.legend()

        plt.grid(alpha=0.3)

        plt.savefig(f'{REPORTS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')

        plt.close()



        print(f"\nROC curve saved to {REPORTS_DIR}")



        return auc_score, f1



    def get_feature_importance(self, top_n=30):

        """Extract and visualize feature importance."""

        print("\n" + "="*60)

        print("FEATURE IMPORTANCE")

        print("="*60)



        importance_df = pd.DataFrame({

            'feature': self.model.feature_name(),

            'importance': self.model.feature_importance(importance_type='gain')

        }).sort_values('importance', ascending=False)



        print(f"\nTop {top_n} Important Features:")

        print(importance_df.head(top_n).to_string())



        # Store for later use

        self.feature_importance = importance_df



        # Visualization

        plt.figure(figsize=(12, 8))

        sns.barplot(

            data=importance_df.head(top_n),

            x='importance',

            y='feature',

            palette='viridis'

        )

        plt.title(f'Top {top_n} Important Features (LightGBM Gain)')

        plt.xlabel('Importance Score')

        plt.tight_layout()

        plt.savefig(f'{REPORTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')

        plt.close()



        print(f"\nFeature importance plot saved to {REPORTS_DIR}")



        return importance_df



    def save_model(self):

        """Save trained model and feature importance."""

        with open(MODEL_PATH, 'wb') as f:

            pickle.dump(self.model, f)



        with open(FEATURE_IMPORTANCE_PATH, 'wb') as f:

            pickle.dump(self.feature_importance, f)



        print(f"\nModel saved to {MODEL_PATH}")

        print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")



    @staticmethod

    def load_model():

        """Load trained model from disk."""

        with open(MODEL_PATH, 'rb') as f:

            model = pickle.load(f)



        print(f"Model loaded from {MODEL_PATH}")

        return model



    @staticmethod

    def load_feature_importance():

        """Load feature importance from disk."""

        with open(FEATURE_IMPORTANCE_PATH, 'rb') as f:

            importance = pickle.load(f)



        return importance



    def run_full_pipeline(self):

        """Execute complete training pipeline."""

        # Load and preprocess

        df = self.load_and_preprocess_data()



        # Split data

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)



        # Apply SMOTE

        X_train_balanced, y_train_balanced = self.apply_smote()



        # Train model

        self.train_model(X_train_balanced, y_train_balanced)



        # Get feature importance

        self.get_feature_importance()



        # Evaluate on validation set

        print("\n--- VALIDATION SET PERFORMANCE ---")

        val_auc, val_f1 = self.evaluate_model(self.X_val, self.y_val)



        # Evaluate on test set

        print("\n--- TEST SET PERFORMANCE ---")

        test_auc, test_f1 = self.evaluate_model(X_test, y_test)



        # Save model and preprocessor

        self.save_model()

        self.preprocessing_pipeline.save_transformers()



        print("\n" + "="*60)

        print("TRAINING COMPLETE")

        print("="*60)



        return {

            'validation_auc': val_auc,

            'test_auc': test_auc,

            'test_f1': test_f1

        }





if __name__ == "__main__":

    trainer = ModelTrainer()

    results = trainer.run_full_pipeline()

    print("\n\nFinal Results:")

    print(results)
