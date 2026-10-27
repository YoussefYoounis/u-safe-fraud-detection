# predict.py



"""



Prediction module.



Contains functions for preprocessing raw input and generating predictions.



"""







import logging

import pandas as pd



import numpy as np



import pickle



from .config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_IMPORTANCE_PATH



from .preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)










class FraudPredictor:



    """Handles predictions on new data."""







    def __init__(self):



        self.model = None



        self.preprocessor = None



        self.feature_importance = None



        self.load_artifacts()







    def load_artifacts(self):



        """Load model and preprocessing artifacts."""



        try:



            with open(MODEL_PATH, 'rb') as f:



                self.model = pickle.load(f)



            logger.info("Model loaded from %s", MODEL_PATH)



        except FileNotFoundError:



            logger.error("Model not found at %s. Run the training pipeline first.", MODEL_PATH)






            raise







        try:



            self.preprocessor = DataPreprocessor.load_transformers()



        except FileNotFoundError:



            logger.error("Preprocessor not found at %s. Run the training pipeline first.", PREPROCESSOR_PATH)






            raise







        try:



            with open(FEATURE_IMPORTANCE_PATH, 'rb') as f:



                self.feature_importance = pickle.load(f)



            logger.info("Feature importance loaded")



        except FileNotFoundError:



            logger.warning("Feature importance not found; explanations will be unavailable")



            self.feature_importance = None











    def create_input_dataframe(self, user_input: dict) -> pd.DataFrame:



        """



        Create a DataFrame from user input, filling missing features with defaults.







        Args:



            user_input: Dictionary with user-provided feature values







        Returns:



            DataFrame ready for preprocessing



        """



        feature_values = dict(user_input)







        # Ensure all expected features are present



        for col in self.preprocessor.feature_names:



            if col not in feature_values:



                # Fill with median/mode from training data



                if col in self.preprocessor.median_values:



                    feature_values[col] = self.preprocessor.median_values[col]



                elif col in self.preprocessor.mode_values:



                    feature_values[col] = self.preprocessor.mode_values[col]



                else:



                    feature_values[col] = 0







        # Keep only the features used during training



        input_df = pd.DataFrame([feature_values])[self.preprocessor.feature_names]







        return input_df







    def preprocess_input(self, user_input: dict) -> pd.DataFrame:



        """



        Preprocess raw user input using fitted transformers.







        Args:



            user_input: Dictionary with user-provided values







        Returns:



            Preprocessed DataFrame



        """



        # Create initial dataframe



        input_df = self.create_input_dataframe(user_input)







        # Apply preprocessing (without fitting)



        input_df_processed = self.preprocessor.preprocess(



            input_df, fit=False, target_col=None



        )







        return input_df_processed







    def predict(self, user_input: dict) -> dict:



        """



        Make prediction on user input.







        Args:



            user_input: Dictionary with user-provided feature values







        Returns:



            Dictionary with prediction results



        """



        try:



            # Preprocess input



            input_processed = self.preprocess_input(user_input)







            # Get prediction



            fraud_probability = self.model.predict(input_processed)[0]



            prediction = "FRAUD" if fraud_probability >= 0.5 else "LEGITIMATE"







            # Get feature importance for this prediction



            top_features = self.get_top_driving_factors(user_input, n=5)







            result = {



                'prediction': prediction,



                'fraud_probability': fraud_probability,



                'legitimate_probability': 1 - fraud_probability,



                'confidence': max(fraud_probability, 1 - fraud_probability),



                'driving_factors': top_features,



                'success': True,



                'error': None



            }







            return result







        except Exception:
            logger.exception("Prediction failed")



            return {



                'prediction': None,



                'fraud_probability': None,



                'legitimate_probability': None,



                'confidence': None,



                'driving_factors': [],



                'success': False,



                'error': 'Prediction could not be completed.'



            }







    def get_top_driving_factors(self, user_input: dict, n: int = 5) -> list:



        """



        Get the top driving factors for a prediction (simplified).



        Returns feature importance features that were provided by user.







        Args:



            user_input: User-provided features



            n: Number of top factors to return







        Returns:



            List of tuples (feature_name, importance_value)



        """



        if self.feature_importance is None:



            return []







        # Get features that were explicitly provided



        user_features = set(user_input.keys())







        # Filter feature importance by user-provided features



        relevant_importance = self.feature_importance[



            self.feature_importance['feature'].isin(user_features)



        ].head(n)







        driving_factors = [



            {



                'feature': row['feature'],



                'importance': float(row['importance']),



                'value': user_input.get(row['feature'], 'N/A')



            }



            for _, row in relevant_importance.iterrows()



        ]







        return driving_factors







    def batch_predict(self, input_df: pd.DataFrame) -> pd.DataFrame:



        """



        Make predictions on multiple inputs at once.







        Args:



            input_df: DataFrame with multiple samples







        Returns:



            DataFrame with predictions



        """



        predictions = []







        for idx, row in input_df.iterrows():



            user_input = row.to_dict()



            result = self.predict(user_input)



            predictions.append(result)







        return pd.DataFrame(predictions)











if __name__ == "__main__":



    # Example usage



    predictor = FraudPredictor()







    # Example input



    test_input = {



        'TransactionAmt': 150.00,



        'card4': 1,  # After encoding



        'card6': 1,



        'DeviceType': 1,



        'P_emaildomain': 0,  # After encoding



        'ProductCD': 0,



        'addr1': 10,



        'dist1': 50,



        'hour_of_day': 14,



        'day_of_week': 2,



        'is_night_transaction': 0,



    }







    result = predictor.predict(test_input)



    print("\nPrediction Result:")



    print(f"Status: {result['prediction']}")



    print(f"Fraud Probability: {result['fraud_probability']:.4f}")



    print(f"Confidence: {result['confidence']:.4f}")



    print(f"Driving Factors: {result['driving_factors']}")
