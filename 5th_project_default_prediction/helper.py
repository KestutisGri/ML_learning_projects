import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from phik import resources, report


def plot_phik_correlation_matrix(df: pd.DataFrame, target_column: str) -> None:
    """
    Plots the Phik correlation matrix for categorical variables including the target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the target column.

    Returns:
    - None (This function creates a visualization and does not return a value)
    """
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_columns.append(target_column)
    categorical_data = df[categorical_columns]

    phik_matrix = categorical_data.phik_matrix(interval_cols=[])

    def get_upper_triangle(matrix: pd.DataFrame) -> pd.DataFrame:
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        return matrix.where(mask)

    upper_triangle_phik_matrix = get_upper_triangle(phik_matrix)
    mask = np.triu(np.ones_like(phik_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        phik_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.8},
        xticklabels=phik_matrix.columns,
        yticklabels=phik_matrix.index,
    )
    plt.title("Phik Correlation Matrix (Lower Triangle)")
    plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from phik import phik_matrix
from multiprocessing import Pool


def calculate_phik(feature_target_pair):
    feature, target = feature_target_pair
    combined_df = pd.DataFrame({"feature": feature, "target": target})
    return (
        feature.name,
        combined_df.phik_matrix(interval_cols=["feature", "target"]).iloc[0, 1],
    )


def plot_phik_correlation_with_target(
    df: pd.DataFrame, target_column: str, remove_columns: list = None, top_n: int = 5
) -> None:
    if remove_columns:
        df = df.drop(columns=remove_columns)

    target_data = df[target_column]

    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.difference(
        [target_column]
    )

    with Pool() as pool:
        correlations = pool.map(
            calculate_phik,
            [
                (df[col], target_data)
                for col in numerical_columns
                if col != target_column
            ],
        )

    phik_dict = {col: corr for col, corr in correlations}
    phik_series = pd.Series(phik_dict)

    abs_target_correlation = phik_series.abs().sort_values(ascending=False)
    top_features = abs_target_correlation.head(top_n).index
    top_correlation = phik_series[top_features]

    sorted_target_correlation = phik_series.sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x=top_correlation.values,
        y=top_correlation.index,
        hue=top_correlation.index,
        palette="coolwarm",
        dodge=False,
        legend=False,
    )
    plt.title(f"Top {top_n} Phik Correlations with {target_column}")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Features")
    plt.show()


def feature_engineering_installments_payments(installments_payments):
    """Processes installments_payments DataFrame to create new features.

    Args:
        installments_payments (pd.DataFrame): The installments_payments DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with new features.
    """

    # Filter and drop columns
    filtered_installments_payments = installments_payments[
        installments_payments["NUM_INSTALMENT_NUMBER"].between(1, 5)
    ]
    columns_to_drop = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    filtered_installments_payments.drop(columns=columns_to_drop, inplace=True)
    installments_payments.drop(columns=columns_to_drop, inplace=True)

    # Define aggregation functions
    agg_funcs = {
        "payment_ratio": ["min", "max", "mean"],
        "DAYS_PAST_DUE": ["min", "max", "mean"],
    }

    # Group and aggregate
    filtered_grouped = filtered_installments_payments.groupby("SK_ID_CURR").agg(
        agg_funcs
    )
    filtered_grouped.columns = [
        "_".join(col).strip() + "_first_five" for col in filtered_grouped.columns
    ]
    grouped = installments_payments.groupby("SK_ID_CURR").agg(agg_funcs)
    grouped.columns = ["_".join(col).strip() + "_all" for col in grouped.columns]

    # Get last values
    last_values = (
        installments_payments.groupby("SK_ID_CURR")
        .agg({"payment_ratio": "last", "DAYS_PAST_DUE": "last"})
        .rename(
            columns={
                "payment_ratio": "payment_ratio_last",
                "DAYS_PAST_DUE": "DAYS_PAST_DUE_last",
            }
        )
    )

    # Merge results
    merged_df = (
        grouped.merge(filtered_grouped, on="SK_ID_CURR", how="left")
        .merge(last_values, on="SK_ID_CURR", how="left")
        .reset_index()
    )

    return merged_df


from typing import List, Dict, Tuple


def clean_dataframe(
    df: pd.DataFrame,
    nan_threshold: float = 0.60,
    same_value_threshold: float = 0.90,
    exceptions: List[str] = [],
) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Cleans a DataFrame by dropping columns with high NaN or constant value ratios.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        nan_threshold (float): The maximum allowed NaN ratio for a column.
        same_value_threshold (float): The maximum allowed ratio of the most common value in a column.
        exceptions (List[str]): A list of column names to exclude from cleaning.

    Returns:
        Tuple[pd.DataFrame, List[str], int]: A tuple containing the cleaned DataFrame, a list of dropped columns, and the total number of columns.
    """
    nan_ratios = df.isna().mean()
    most_common_value_ratios = df.apply(
        lambda col: col.value_counts(normalize=True).max()
    )
    columns_to_drop = df.columns[
        (nan_ratios > nan_threshold) | (most_common_value_ratios > same_value_threshold)
    ]
    columns_to_drop = [col for col in columns_to_drop if col not in exceptions]
    return df.drop(columns=columns_to_drop), list(columns_to_drop), df.shape[1]


def clean_all_dataframes(
    dataframes: Dict[str, pd.DataFrame],
    nan_threshold: float = 0.60,
    same_value_threshold: float = 0.90,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, int]]:
    """
    Cleans multiple DataFrames by dropping columns with high NaN or constant value ratios.

    Args:
        dataframes (Dict[str, pd.DataFrame]): A dictionary of DataFrames to clean.
        nan_threshold (float): The maximum allowed NaN ratio for a column.
        same_value_threshold (float): The maximum allowed ratio of the most common value in a column.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, int]]: A tuple containing the cleaned DataFrames, a dictionary of dropped columns, and a dictionary of total columns.
    """
    dropped_columns_dict = {}
    total_columns_dict = {}

    for name, df in dataframes.items():
        exceptions = ["TARGET"] if "TARGET" in df.columns else []
        cleaned_df, dropped_columns, total_columns = clean_dataframe(
            df, nan_threshold, same_value_threshold, exceptions
        )
        dataframes[name] = cleaned_df
        dropped_columns_dict[name] = dropped_columns
        total_columns_dict[name] = total_columns

        # Save dropped columns to CSV and display summary
        pd.DataFrame(dropped_columns, columns=["Dropped Columns"]).to_csv(
            f"dropped_columns_{name}.csv", index=False
        )
        print(f"Cleaned DataFrame: {name}")
        print(f"Dropped Columns: {dropped_columns}")
        print(f"Total Columns: {total_columns}")
        print(f"Columns Dropped: {len(dropped_columns)}")

    return dataframes, dropped_columns_dict, total_columns_dict


import pandas as pd
from typing import Dict, Tuple


def calculate_common_id_percentages(
    application_train: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> Tuple[Dict[str, float], float]:
    """
    Calculate the percentage of common SK_ID_CURR and SK_ID_BUREAU values between datasets.

    Args:
        application_train (pd.DataFrame): The application_train DataFrame containing SK_ID_CURR.
        datasets (Dict[str, pd.DataFrame]): A dictionary of DataFrames to compare SK_ID_CURR.
        bureau (pd.DataFrame): The bureau DataFrame containing SK_ID_BUREAU.
        bureau_balance (pd.DataFrame): The bureau_balance DataFrame containing SK_ID_BUREAU.

    Returns:
        Tuple[Dict[str, float], float]: A tuple containing a dictionary of percentages for SK_ID_CURR and a percentage for SK_ID_BUREAU.
    """
    sk_id_curr_application = application_train["SK_ID_CURR"]

    def calculate_percentage_common(
        df: pd.DataFrame, sk_id_curr_application: pd.Series
    ) -> float:
        sk_id_curr_df = df["SK_ID_CURR"]
        common_sk_id_curr = sk_id_curr_application[
            sk_id_curr_application.isin(sk_id_curr_df)
        ]
        percentage_common = (len(common_sk_id_curr) / len(sk_id_curr_application)) * 100
        return percentage_common

    percentages_sk_id_curr = {
        name: calculate_percentage_common(df, sk_id_curr_application)
        for name, df in datasets.items()
    }

    sk_id_bureau = bureau["SK_ID_BUREAU"]
    sk_id_bureau_balance = bureau_balance["SK_ID_BUREAU"]
    common_sk_id_bureau = sk_id_bureau[sk_id_bureau.isin(sk_id_bureau_balance)]
    percentage_sk_id_bureau = (len(common_sk_id_bureau) / len(sk_id_bureau)) * 100

    return percentages_sk_id_curr, percentage_sk_id_bureau


import pandas as pd
import numpy as np
from typing import List


def process_previous_application(previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    Process the previous_application DataFrame to create new features.

    This function performs the following steps:
    1. Identifies categorical features and counts occurrences of each value grouped by SK_ID_CURR.
    2. Identifies numerical features and calculates mean, max, and min values grouped by SK_ID_CURR.
    3. Extracts the last row for each SK_ID_CURR.
    4. Merges the categorical counts, numerical aggregations, and last row values into a single DataFrame.

    Parameters:
    previous_application (pd.DataFrame): The input DataFrame containing previous application data.

    Returns:
    pd.DataFrame: The processed DataFrame with new features.
    """
    # Identify categorical features
    categorical_features: List[str] = previous_application.select_dtypes(
        include=[object]
    ).columns.tolist()

    # Initialize an empty list to store the DataFrames
    categorical_counts_list: List[pd.DataFrame] = []

    # Loop through each categorical feature
    for col in categorical_features:
        # Group by SK_ID_CURR and count occurrences of each value in the categorical column
        counts = (
            previous_application.groupby("SK_ID_CURR")[col]
            .value_counts()
            .unstack(fill_value=0)
        )
        # Rename columns to include the feature name
        counts.columns = [f"{col}_{val}" for val in counts.columns]
        # Append the DataFrame to the list
        categorical_counts_list.append(counts)

    # Concatenate all the DataFrames
    categorical_counts = pd.concat(categorical_counts_list, axis=1)

    # Identify numerical features excluding SK_ID_CURR and SK_ID_PREV
    numerical_features: List[str] = (
        previous_application.select_dtypes(include=[np.number])
        .columns.difference(["SK_ID_CURR", "SK_ID_PREV"])
        .tolist()
    )

    # Sort the DataFrame by SK_ID_CURR and SK_ID_PREV
    previous_application_sorted = previous_application.sort_values(
        by=["SK_ID_CURR", "SK_ID_PREV"]
    )

    # Get the last row for each SK_ID_CURR
    last_rows = previous_application_sorted.groupby("SK_ID_CURR").last().reset_index()

    # Group by SK_ID_CURR and calculate mean, max, and min for each numerical column from the entire DataFrame
    numerical_aggregations = previous_application.groupby("SK_ID_CURR")[
        numerical_features
    ].agg(["mean", "max", "min"])

    # Flatten the MultiIndex columns
    numerical_aggregations.columns = [
        "_".join(col).strip() for col in numerical_aggregations.columns.values
    ]

    # Merge categorical counts and numerical aggregations
    merged_previous_application = categorical_counts.merge(
        numerical_aggregations, on="SK_ID_CURR", how="left"
    )

    # Merge the last row values (both numerical and categorical)
    last_row_columns = ["SK_ID_CURR"] + numerical_features + categorical_features
    merged_previous_application = merged_previous_application.merge(
        last_rows[last_row_columns], on="SK_ID_CURR", how="left", suffixes=("", "_last")
    )

    return merged_previous_application
