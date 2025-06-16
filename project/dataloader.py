import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler


class DataLoader:
    def __init__(self, data_path: str, random_state: int = 1):
        self.data_path = data_path
        self.random_state = random_state
        self.raw_data = None
        self.processed_data = None

        v_columns = [f"V{i}" for i in range(1, 29)]
        passthrough_columns = v_columns + ["Class"]

        self.preprocessing_pipeline = ColumnTransformer(
            [
                ("time_scaler", MinMaxScaler(), ["Time"]),
                ("amount_scaler", RobustScaler(), ["Amount"]),
                ("passthrough", "passthrough", passthrough_columns),
            ]
        )

    def load_data(self) -> pd.DataFrame:
        self.raw_data = pd.read_csv(self.data_path)
        return self.raw_data

    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if data is None:
            if self.raw_data is None:
                raise ValueError(
                    "No data loaded. Call load_data() first or provide data parameter."
                )
            data = self.raw_data.copy()

        processed_array = self.preprocessing_pipeline.fit_transform(data)
        self.processed_data = pd.DataFrame(processed_array, columns=data.columns)

        self.processed_data = self.processed_data.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)

        return self.processed_data

    def split_data(
        self,
        data: Optional[pd.DataFrame] = None,
        train_size: int = 240000,
        test_size: int = 22000,
        val_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if data is None:
            if self.processed_data is None:
                raise ValueError(
                    "No processed data available. Call preprocess_data() first."
                )
            data = self.processed_data

        total_size = len(data)

        if val_size is None:
            val_size = total_size - train_size - test_size

        if train_size + test_size + val_size > total_size:
            raise ValueError(
                f"Split sizes ({train_size} + {test_size} + {val_size}) exceed data size ({total_size})"
            )

        train_end = train_size
        test_end = train_size + test_size

        train = data[:train_end].copy()
        test = data[train_end:test_end].copy()
        val = data[test_end : test_end + val_size].copy()

        return train, test, val

    def create_balanced_dataset(
        self, data: Optional[pd.DataFrame] = None, target_col: str = "Class"
    ) -> pd.DataFrame:
        """
        Create a balanced dataset by undersampling the majority class.
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError(
                    "No processed data available. Call preprocess_data() first."
                )
            data = self.processed_data

        frauds = data.query(f"{target_col} == 1")
        not_frauds = data.query(f"{target_col} == 0")

        balanced_not_frauds = not_frauds.sample(
            len(frauds) * 2, random_state=self.random_state
        )

        balanced_df = pd.concat([frauds, balanced_not_frauds])
        balanced_df = balanced_df.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)

        return balanced_df

    def split_balanced_data(
        self,
        balanced_data: pd.DataFrame,
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:

        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        total_size = len(balanced_data)
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)

        # Convert to numpy
        data_np = balanced_data.to_numpy()

        # Split the data
        train_data = data_np[:train_size]
        test_data = data_np[train_size : train_size + test_size]
        val_data = data_np[train_size + test_size :]

        # Separate features and targets
        X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
        X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)
        X_val, y_val = val_data[:, :-1], val_data[:, -1].astype(int)

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    def get_numpy_splits(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        train_np = train_df.to_numpy()
        test_np = test_df.to_numpy()
        val_np = val_df.to_numpy()

        # Separate features and targets (assuming target is last column)
        X_train, y_train = train_np[:, :-1], train_np[:, -1]
        X_test, y_test = test_np[:, :-1], test_np[:, -1]
        X_val, y_val = val_np[:, :-1], val_np[:, -1]

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    def load_and_preprocess(self) -> pd.DataFrame:
        return self.preprocess_data(self.load_data())

    def get_feature_names(self) -> list:
        return self.preprocessing_pipeline.get_feature_names_out()
