"""
Winsorization Implementation for Academic Research
---------------------------------------------------
This module provides a class for performing winsorizations, including MAD, k-sigma, 
and winsorization at extreme percentiles. The implementation focuses on clarity, 
documentation, and best practices for financial research.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple

class Winsorizer:
    """
    A class to perform winsorizations on cross-sectional characteristic matrices.
    Supports input as either pandas DataFrame or numpy array.
    """

    @staticmethod
    def __to_dataframe(features: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Convert the input features to a pandas DataFrame if it is a numpy array.

        Args:
            features (Union[pd.DataFrame, np.ndarray]): The input features.

        Returns:
            pd.DataFrame: The input converted to a DataFrame.
        """
        if isinstance(features, pd.DataFrame):
            return features.copy()
        elif isinstance(features, np.ndarray):
            return pd.DataFrame(features)
        else:
            raise TypeError("Input features must be a pandas DataFrame or a numpy array.")

    @staticmethod
    def __to_original_type(result: pd.DataFrame, original: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Convert the DataFrame result back to the type of the original input.

        Args:
            result (pd.DataFrame): The processed DataFrame.
            original (Union[pd.DataFrame, np.ndarray]): The original input features.

        Returns:
            Union[pd.DataFrame, np.ndarray]: The result in the same type as the original input.
        """
        if isinstance(original, np.ndarray):
            return result.values
        return result

    @classmethod
    def MAD_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        scaled: bool = False,
        k: int = 3
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the Median Absolute Deviation (MAD) method.

        Args:
            features (Union[pd.DataFrame, np.ndarray]): Cross-sectional characteristic matrix, 
                where each column represents a feature.
            scaled (bool, optional): Whether to scale the MAD value (MAD * 1.4826). Default is False.
            k (int, optional): Scaling factor to determine limits. Default is 3.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using MAD.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate the median for each column
        median = df.median()
        # Compute the absolute deviation from the median, then calculate the median of these deviations for each column
        mad = (df - median).abs().median()

        # Scale the MAD if required
        if scaled:
            mad *= 1.4826

        # Calculate the lower and upper limits for winsorization
        lower = median - k * mad
        upper = median + k * mad

        # Apply winsorization using the DataFrame.clip() method
        result = df.clip(lower=lower, upper=upper, axis=1)
        return cls.__to_original_type(result, original)

    @classmethod
    def sigma_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        k: int = 3
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the k-sigma rule.

        Args:
            features (Union[pd.DataFrame, np.ndarray]): Cross-sectional characteristic matrix, 
                where each column represents a feature.
            k (int, optional): Scaling factor to determine limits. Default is 3.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using the k-sigma rule.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate the mean and standard deviation for each column
        mean = df.mean()
        std = df.std()

        # Calculate the lower and upper limits for winsorization
        lower = mean - k * std
        upper = mean + k * std

        # Apply winsorization using the DataFrame.clip() method
        result = df.clip(lower=lower, upper=upper, axis=1)
        return cls.__to_original_type(result, original)

    @classmethod
    def percentile_winsorization(
        cls,
        features: Union[pd.DataFrame, np.ndarray],
        percentile: Tuple[float, float] = (0.01, 0.99),
        drop_outlier: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply winsorization on features using the percentile rule.

        Args:
            features (Union[pd.DataFrame, np.ndarray]): Cross-sectional characteristic matrix, 
                where each column represents a feature.
            percentile (Tuple[float, float], optional): The lower and upper percentiles to winsorize. 
                Default is (0.01, 0.99).
            drop_outlier (bool, optional): Whether to drop outliers instead of clipping them. Default is False.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Winsorized features using the percentile rule.
        """
        original = features
        df = cls.__to_dataframe(features)

        # Calculate lower and upper bounds based on the given percentiles
        lower_bound = df.quantile(percentile[0])
        upper_bound = df.quantile(percentile[1])

        if drop_outlier:
            # Drop rows with any outlier values: True if all values are within [lower_bound, upper_bound]
            mask = (df >= lower_bound) & (df <= upper_bound)
            result = df[mask.all(axis=1)]
        else:
            # Clip values to the specified bounds
            result = df.clip(lower=lower_bound, upper=upper_bound, axis=1)

        return cls.__to_original_type(result, original)