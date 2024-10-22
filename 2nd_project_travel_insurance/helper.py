import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def detect_outliers(df, cols_to_check):
    """
    Detects outliers in the specified columns of a DataFrame using the IQR method.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check for outliers.
    cols_to_check (list): A list of column names to check for outliers.

    Returns:
    None
    """
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        print(f"Number of outliers in {col}: {len(outliers)}")


def plot_outliers(df, cols_to_check):
    """
    Creates box plots for the specified columns of a DataFrame to visualize outliers.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    cols_to_check (list): A list of column names to plot.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    for i, col in enumerate(cols_to_check):
        plt.subplot(1, len(cols_to_check), i + 1)
        sns.boxplot(y=df[col])
        plt.title(f'Box plot of {col}')

    plt.tight_layout()
    plt.show()


def plot_data(df, plot_specs):
    """
    Plots the specified columns of a DataFrame according to the given specifications.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    plot_specs (list of tuples): A list of tuples, where each tuple contains the column name and the plot type.

    Returns:
    None
    """
    fig, axs = plt.subplots(len(plot_specs) // 2, 2, figsize=(20, 12))

    for i, (col, plot_type) in enumerate(plot_specs):
        if plot_type == 'hist':
            sns.histplot(df[col], kde=True, ax=axs[i // 2, i % 2])
        elif plot_type == 'count':
            ax = sns.countplot(x=col, data=df, ax=axs[i // 2, i % 2])
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10),
                            textcoords='offset points')
            heights = [p.get_height() for p in ax.patches]
            max_height = max(heights)
            min_height = min(heights)
            for p in ax.patches:
                if p.get_height() == max_height:
                    p.set_color('green')
                elif p.get_height() == min_height:
                    p.set_color('red')
        axs[i // 2, i % 2].set_title(f'{col} {plot_type}')

    plt.tight_layout()
    plt.show()


def plot_phik(df, id_column=None):
    """
    Computes the Phi-K correlation matrix for a DataFrame and plots it as a heatmap.

    Parameters:
    df (pandas.DataFrame): The DataFrame to compute the Phi-K correlation matrix for.
    id_column (str, optional): The name of the ID column to drop from the DataFrame. Defaults to None.

    Returns:
    None
    """
    if id_column:
        df = df.drop(columns=[id_column])

    phi_k_matrix = df.phik_matrix()
    sns.heatmap(phi_k_matrix, annot=True)
    plt.show()


def generate_classification_reports(models, X_train, y_train, X_val, y_val, train_model=True):
    """
    Trains each model in the provided dictionary on the training data (if specified),
    makes predictions on the validation data,
    generates a classification report for each model, and compiles these reports into a single DataFrame.

    Parameters:
    models (dict): A dictionary where the keys are model names (str) and the values are sklearn model instances.
    X_train (pandas.DataFrame): The training data.
    y_train (pandas.Series): The labels for the training data.
    X_val (pandas.DataFrame): The validation data.
    y_val (pandas.Series): The labels for the validation data.
    train_model (bool): Whether to train the model or not. Defaults to True.

    Returns:
    final_report_df (pandas.DataFrame): A DataFrame where each row corresponds to a class-label and each model's
                                        performance metrics for that class. The 'Model' column indicates the model.
    """
    classification_reports = []

    for name, model in models.items():
        if train_model:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)  # Use validation set for predictions

        report_dict = classification_report(y_val, y_pred,
                                            output_dict=True)  # Compare predictions to validation set labels
        report_df = pd.DataFrame(report_dict).transpose()

        report_df['Model'] = name

        classification_reports.append(report_df)

    final_report_df = pd.concat(classification_reports)

    final_report_df.reset_index(inplace=True)

    final_report_df.rename(columns={'index': 'Class'}, inplace=True)

    return final_report_df


def prepare_data(df, drop_columns, target_column, test_size, val_size, random_state, stratify=None):
    """
    Splits the provided DataFrame into training, validation, and test sets.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.
    drop_columns (list): The list of columns to drop.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the test dataset to include in the validation split.
    random_state (int): The seed used by the random number generator.
    stratify (array-like): If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
    X_train (pandas.DataFrame): The training data.
    X_val (pandas.DataFrame): The validation data.
    X_test (pandas.DataFrame): The test data.
    y_train (pandas.Series): The labels for the training data.
    y_val (pandas.Series): The labels for the validation data.
    y_test (pandas.Series): The labels for the test data.
    """
    features = df.drop(drop_columns, axis=1)
    target = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=test_size,
                                                        stratify=target if stratify else None,
                                                        random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size,
                                                    stratify=y_temp if stratify else None, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_confusion_matrix(y_true, y_pred, title=None):
    """
    Plots a confusion matrix using seaborn's heatmap.

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels of the data.
    title (str, optional): Title for the heatmap. Defaults to None.

    Returns:
    None
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    if title:
        plt.title(title)
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name):
    # Calculate the probabilities of the predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Calculate the AUC score
    auc_score = auc(fpr, tpr)

    # Identify the best threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Best threshold: {optimal_threshold}")

    # Plot the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC ({model_name}) = {auc_score:.2f}")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Best Threshold')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')  # Line of no-discrimination
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    plt.grid()
    plt.show()
