# data_loader.py

"""

Data loading and merging module.

Handles loading transaction and identity data from CSV files.

"""



import pandas as pd

from .config import (

    TRAIN_TRANSACTION_PATH,

    TRAIN_IDENTITY_PATH,

    TEST_TRANSACTION_PATH

)





class DataLoader:

    """Loads and merges transaction and identity data."""



    def __init__(self):

        self.train_transaction = None

        self.train_identity = None

        self.test_transaction = None

        self.test_identity = None



    def load_train_data(self):

        """Load training transaction and identity data."""

        print("Loading training transaction data...")

        self.train_transaction = pd.read_csv(TRAIN_TRANSACTION_PATH)



        print("Loading training identity data...")

        self.train_identity = pd.read_csv(TRAIN_IDENTITY_PATH)



        print(f"Train transactions shape: {self.train_transaction.shape}")

        print(f"Train identity shape: {self.train_identity.shape}")



        return self.train_transaction, self.train_identity



    def load_test_data(self):

        """Load test transaction data (identity optional)."""

        print("Loading test transaction data...")

        self.test_transaction = pd.read_csv(TEST_TRANSACTION_PATH)



        print(f"Test transactions shape: {self.test_transaction.shape}")



        return self.test_transaction



    def merge_data(self, transaction_df, identity_df):

        """

        Merge transaction and identity data.



        Args:

            transaction_df: DataFrame with transaction data

            identity_df: DataFrame with identity data



        Returns:

            Merged DataFrame

        """

        merged = transaction_df.merge(

            identity_df,

            on='TransactionID',

            how='left'

        )



        print(f"Merged data shape: {merged.shape}")

        print(f"Missing values after merge:\n{merged.isnull().sum().head(10)}")



        return merged



    def get_merged_train_data(self):

        """Load and merge train data in one call."""

        if self.train_transaction is None:

            self.load_train_data()



        return self.merge_data(self.train_transaction, self.train_identity)



    def get_data_info(self):

        """Print comprehensive data information."""

        if self.train_transaction is not None:

            print("\n=== Transaction Data Info ===")

            print(self.train_transaction.info())

            print("\nTransaction Data Sample:")

            print(self.train_transaction.head())



        if self.train_identity is not None:

            print("\n=== Identity Data Info ===")

            print(self.train_identity.info())

            print("\nIdentity Data Sample:")

            print(self.train_identity.head())





if __name__ == "__main__":

    loader = DataLoader()

    train_df = loader.get_merged_train_data()

    print(train_df.head())
