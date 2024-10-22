from typing import List, Optional, Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import joblib


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for encoding multiple columns in a DataFrame using Label Encoding.

    Attributes:
        columns (Optional[List[str]]): List of column names to encode.
        encoders (Dict[str, LabelEncoder]): Dictionary of LabelEncoders for each column.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initializes the MultiColumnLabelEncoder with the specified columns.

        Args:
            columns (Optional[List[str]]): List of column names to encode. If None, no columns will be encoded.
        """
        self.columns = columns
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(
            self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "MultiColumnLabelEncoder":
        """
        Fits the LabelEncoders to the specified columns in the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Optional[pd.Series]): Ignored. This parameter exists for compatibility with scikit-learn's fit method.

        Returns:
            MultiColumnLabelEncoder: The fitted transformer.
        """
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the specified columns in the DataFrame using the fitted LabelEncoders.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame with encoded columns.
        """
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy

    def get_feature_names_out(
            self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Returns the names of the features after transformation.

        Args:
            input_features (Optional[List[str]]): Ignored. This parameter exists for compatibility with scikit-learn's get_feature_names_out method.

        Returns:
            List[str]: The list of column names.
        """
        return self.columns


from typing import List
import re


def clean_feature_names(feature_names: List[str]) -> List[str]:
    """
    Cleans a list of feature names by removing any characters that are not alphanumeric or underscores.

    Args:
        feature_names (List[str]): A list of feature names to be cleaned.

    Returns:
        List[str]: A list of cleaned feature names.
    """
    pattern = re.compile(r"[^a-zA-Z0-9_]")
    return [pattern.sub("", name) for name in feature_names]


import pandas as pd


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that all columns in the DataFrame are numeric and cleans the column names.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with all columns converted to numeric and cleaned column names.
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.columns = clean_feature_names(df.columns)
    return df


from typing import Tuple, List
from typing import Tuple
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calculate_metrics(
        y_true: pd.Series, probs_original: pd.Series, probs_calibrated: pd.Series
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate Brier scores, log loss, and ROC AUC for original and calibrated models.

    Args:
        y_true (pd.Series): True labels.
        probs_original (pd.Series): Predicted probabilities from the original model.
        probs_calibrated (pd.Series): Predicted probabilities from the calibrated model.

    Returns:
        Tuple[float, float, float, float, float, float]: Brier scores, log loss, and ROC AUC for original and calibrated models.
    """
    brier_original = brier_score_loss(y_true, probs_original)
    brier_calibrated = brier_score_loss(y_true, probs_calibrated)
    log_loss_original = log_loss(y_true, probs_original)
    log_loss_calibrated = log_loss(y_true, probs_calibrated)
    roc_auc_original = roc_auc_score(y_true, probs_original)
    roc_auc_calibrated = roc_auc_score(y_true, probs_calibrated)

    return (
        brier_original,
        brier_calibrated,
        log_loss_original,
        log_loss_calibrated,
        roc_auc_original,
        roc_auc_calibrated,
    )


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curves(
        y_true: pd.Series, probs_original: pd.Series, probs_calibrated: pd.Series
) -> None:
    """
    Plot calibration curves for original and calibrated models.

    Args:
        y_true (pd.Series): True labels.
        probs_original (pd.Series): Predicted probabilities from the original model.
        probs_calibrated (pd.Series): Predicted probabilities from the calibrated model.
    """
    plt.figure(figsize=(10, 5))

    # Calibration curve for the original model
    prob_true, prob_pred = calibration_curve(y_true, probs_original, n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o", label="Original Model")

    # Calibration curve for the calibrated model
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, probs_calibrated, n_bins=10
    )
    plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="Calibrated Model")

    # Plot settings
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves")
    plt.legend()
    plt.show()


from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_confusion_matrices(
        y_true: pd.Series,
        probs_original: np.ndarray,
        probs_calibrated: np.ndarray,
        threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate confusion matrices for original and calibrated models.

    Args:
        y_true (pd.Series): True labels.
        probs_original (np.ndarray): Predicted probabilities from the original model.
        probs_calibrated (np.ndarray): Predicted probabilities from the calibrated model.
        threshold (float): Threshold to convert probabilities to binary predictions. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Confusion matrices for original and calibrated models.
    """
    preds_original = (probs_original >= threshold).astype(int)
    preds_calibrated = (probs_calibrated >= threshold).astype(int)

    conf_matrix_original = confusion_matrix(y_true, preds_original)
    conf_matrix_calibrated = confusion_matrix(y_true, preds_calibrated)

    return conf_matrix_original, conf_matrix_calibrated


def plot_probability_distributions(
        probs_original: np.ndarray, probs_calibrated: np.ndarray
) -> None:
    """
    Plot probability distributions for original and calibrated models.

    Args:
        probs_original (np.ndarray): Predicted probabilities from the original model.
        probs_calibrated (np.ndarray): Predicted probabilities from the calibrated model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(probs_original, bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Probability Distribution - Original Model")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Frequency")

    sns.histplot(probs_calibrated, bins=20, kde=True, ax=axes[1])
    axes[1].set_title("Probability Distribution - Calibrated Model")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
        conf_matrix_original: np.ndarray, conf_matrix_calibrated: np.ndarray
) -> None:
    """
    Plot confusion matrices for original and calibrated models.

    Args:
        conf_matrix_original (np.ndarray): Confusion matrix for the original model.
        conf_matrix_calibrated (np.ndarray): Confusion matrix for the calibrated model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(conf_matrix_original, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix - Original Model")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(conf_matrix_calibrated, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title("Confusion Matrix - Calibrated Model")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


import pandas as pd
from typing import Tuple


def analyze_credit_predictions(
        df: pd.DataFrame,
        amt_credit_col: str,
        model,
        X_test_transformed: pd.DataFrame,
        y_test: pd.Series,
) -> pd.DataFrame:
    """
    Analyze credit predictions by categorizing AMT_CREDIT into groups and calculating prediction accuracy.

    Args:
        df (pd.DataFrame): The original DataFrame containing the data.
        amt_credit_col (str): The column name for AMT_CREDIT.
        model: The trained model used for predictions.
        X_test_transformed (pd.DataFrame): The transformed test set features.
        y_test (pd.Series): The true labels for the test set.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results.
    """

    def categorize_credit(amt_credit):
        if amt_credit <= 100000:
            return "0-100k"
        elif amt_credit <= 500000:
            return "100k-500k"
        elif amt_credit <= 1000000:
            return "500k-1mln"
        else:
            return "1mln+"

    df["CREDIT_GROUP"] = df[amt_credit_col].apply(categorize_credit)

    # Generate predictions
    preds = model.predict(X_test_transformed)

    # Calculate the number of samples and the number of wrongly predicted samples for each group
    results = df.loc[X_test_transformed.index].copy()
    results["PREDICTION"] = preds
    results["ACTUAL"] = y_test.values
    results["WRONG_PREDICTION"] = results["PREDICTION"] != results["ACTUAL"]

    grouped_results = results.groupby("CREDIT_GROUP").agg(
        total_samples=("WRONG_PREDICTION", "size"),
        wrongly_predicted=("WRONG_PREDICTION", "sum"),
    )

    # Calculate the percentage of wrongly predicted samples for each group
    grouped_results["wrongly_predicted_percentage"] = (
                                                              grouped_results["wrongly_predicted"] / grouped_results[
                                                          "total_samples"]
                                                      ) * 100

    # Create a new DataFrame to store the results
    grouped_results.reset_index(inplace=True)
    grouped_results.rename(
        columns={
            "CREDIT_GROUP": "Credit Group",
            "total_samples": "Total Samples",
            "wrongly_predicted": "Wrongly Predicted",
            "wrongly_predicted_percentage": "Wrongly Predicted (%)",
        },
        inplace=True,
    )

    return grouped_results
