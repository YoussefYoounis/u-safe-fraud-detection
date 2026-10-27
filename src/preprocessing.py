# preprocessing.py

"""

Data preprocessing module.

Handles missing value imputation, categorical encoding, scaling, and feature engineering.

"""



import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.impute import SimpleImputer

import pickle

from .config import SCALER_PATH, PREPROCESSOR_PATH, RANDOM_STATE





class DataPreprocessor:

    """Handles all data preprocessing tasks."""



    def __init__(self):

        self.scaler = StandardScaler()

        self.label_encoders = {}

        self.numerical_imputer = SimpleImputer(strategy='median')

        self.categorical_imputer = SimpleImputer(strategy='most_frequent')

        self.numerical_features = None

        self.categorical_features = None

        self.median_values = {}

        self.mode_values = {}

        self.feature_names = None



    def identify_feature_types(self, df):

        """Identify numerical and categorical features."""

        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()



        # Remove target variable

        if 'isFraud' in self.numerical_features:

            self.numerical_features.remove('isFraud')



        print(f"Numerical features ({len(self.numerical_features)}): {self.numerical_features[:10]}...")

        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")



        return self.numerical_features, self.categorical_features



    def handle_missing_values(self, df, fit=False):

        """Handle missing values in numerical and categorical features."""

        df = df.copy()



        if fit:

            print("Fitting imputers on training data...")

            # Fit imputers

            self.numerical_imputer.fit(df[self.numerical_features])

            self.categorical_imputer.fit(df[self.categorical_features])



            # Store median and mode for later reference

            for col in self.numerical_features:

                self.median_values[col] = df[col].median()



            for col in self.categorical_features:

                self.mode_values[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'



        # Transform

        df[self.numerical_features] = self.numerical_imputer.transform(df[self.numerical_features])

        df[self.categorical_features] = self.categorical_imputer.transform(df[self.categorical_features])



        print(f"Missing values after imputation: {df.isnull().sum().sum()}")



        return df



    def encode_categorical_features(self, df, fit=False):

        """Encode categorical features using LabelEncoder."""

        df = df.copy()



        for col in self.categorical_features:

            if fit:

                self.label_encoders[col] = LabelEncoder()

                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

            else:

                # Handle unseen categories

                df[col] = df[col].astype(str)

                known_classes = set(self.label_encoders[col].classes_)

                df[col] = df[col].map(

                    lambda x: self.label_encoders[col].transform([x])[0]

                    if x in known_classes else -1

                )



        print("Categorical encoding completed.")



        return df



    def scale_numerical_features(self, df, fit=False):

        """Scale numerical features using StandardScaler."""

        df = df.copy()



        if fit:

            print("Fitting scaler on training data...")

            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])

        else:

            df[self.numerical_features] = self.scaler.transform(df[self.numerical_features])



        print("Scaling completed.")



        return df



    def engineer_time_features(self, df):

        """Create time-based features from TransactionDT."""

        df = df.copy()



        time_features = {'hour_of_day', 'day_of_week', 'is_night_transaction'}
        if 'TransactionDT' in df.columns and not time_features.issubset(df.columns):

            # Convert TransactionDT to hours and extract patterns

            # TransactionDT is seconds from a reference point

            df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24

            df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

            df['is_night_transaction'] = ((df['hour_of_day'] >= 22) |

                                         (df['hour_of_day'] < 6)).astype(int)



            print("Time features engineered: hour_of_day, day_of_week, is_night_transaction")



        return df



    def preprocess(self, df, fit=False, target_col='isFraud'):

        """

        Complete preprocessing pipeline.



        Args:

            df: Input DataFrame

            fit: Whether to fit the transformers (True for training, False for test)

            target_col: Name of target column



        Returns:

            Preprocessed DataFrame

        """

        print("\n" + "="*60)

        print("STARTING DATA PREPROCESSING")

        print("="*60)





        # Separate target if it exists

        target = None

        if target_col in df.columns:

            target = df[[target_col]].copy()

            df = df.drop(columns=[target_col])



        # Identify feature types

        if fit:

            self.identify_feature_types(df)



        # Engineer time features

        df = self.engineer_time_features(df)



        # Handle missing values

        df = self.handle_missing_values(df, fit=fit)



        # Encode categorical features

        df = self.encode_categorical_features(df, fit=fit)



        # Scale numerical features

        df = self.scale_numerical_features(df, fit=fit)



        # Store feature names

        if fit:

            self.feature_names = df.columns.tolist()



        # Reattach target if it exists

        if target is not None:

            df = pd.concat([df, target], axis=1)



        print("="*60)

        print("PREPROCESSING COMPLETE")

        print("="*60 + "\n")



        return df



    def save_transformers(self):

        """Save fitted transformers to disk."""

        preprocessor_state = {

            'scaler': self.scaler,

            'label_encoders': self.label_encoders,

            'numerical_imputer': self.numerical_imputer,

            'categorical_imputer': self.categorical_imputer,

            'numerical_features': self.numerical_features,

            'categorical_features': self.categorical_features,

            'median_values': self.median_values,

            'mode_values': self.mode_values,

            'feature_names': self.feature_names,

        }



        with open(PREPROCESSOR_PATH, 'wb') as f:

            pickle.dump(preprocessor_state, f)



        print(f"Preprocessor saved to {PREPROCESSOR_PATH}")



    @staticmethod

    def load_transformers():

        """Load fitted transformers from disk."""

        with open(PREPROCESSOR_PATH, 'rb') as f:

            state = pickle.load(f)



        preprocessor = DataPreprocessor()

        preprocessor.scaler = state['scaler']

        preprocessor.label_encoders = state['label_encoders']

        preprocessor.numerical_imputer = state['numerical_imputer']

        preprocessor.categorical_imputer = state['categorical_imputer']

        preprocessor.numerical_features = state['numerical_features']

        preprocessor.categorical_features = state['categorical_features']

        preprocessor.median_values = state['median_values']

        preprocessor.mode_values = state['mode_values']

        preprocessor.feature_names = state['feature_names']



        print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")



        return preprocessor





if __name__ == "__main__":

    from .data_loader import DataLoader



    loader = DataLoader()

    train_df = loader.get_merged_train_data()



    preprocessor = DataPreprocessor()

    train_df_processed = preprocessor.preprocess(train_df, fit=True)

    print(train_df_processed.head())

    print(f"Processed shape: {train_df_processed.shape}")
