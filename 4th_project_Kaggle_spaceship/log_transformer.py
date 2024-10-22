from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for applying a logarithmic transformation to specified columns of a DataFrame,
    with handling for NaN values by temporarily filling them with a placeholder value.

    Parameters:
    columns (list): The list of column names to apply the transformation to.
    placeholder_value (float): The value to temporarily replace NaNs with before applying the log transformation.

    Returns:
    DataFrame: The transformed DataFrame.
    """

    def __init__(self, columns, placeholder_value=1):
        """
        Initialize the transformer with the list of columns to transform and the placeholder value.

        Parameters:
        columns (list): The list of column names to apply the transformation to.
        placeholder_value (float): The value to temporarily replace NaNs with before applying the log transformation.
        """
        self.columns = columns
        self.placeholder_value = placeholder_value

    def fit(self, X, y=None):
        """
        Fit the transformer. This function does nothing as no fitting is necessary.

        Parameters:
        X (DataFrame): The DataFrame to transform.
        y (Series, optional): The target variable. Not used.

        Returns:
        self
        """
        return self

    def transform(self, X):
        """
        Apply the logarithmic transformation to the specified columns of the DataFrame,
        handling NaN values by temporarily filling them with a placeholder value.

        Parameters:
        X (DataFrame): The DataFrame to transform.

        Returns:
        DataFrame: The transformed DataFrame.
        """
        X_copy = X.copy()
        for col in self.columns:
            # Temporarily fill NaN values with the placeholder value
            X_copy[col] = X_copy[col].fillna(self.placeholder_value)
            # Apply the logarithmic transformation
            X_copy[col] = np.log1p(X_copy[col])
            # Revert the placeholder values back to NaN
            X_copy[col] = X_copy[col].replace(np.log1p(self.placeholder_value), np.nan)
        return X_copy