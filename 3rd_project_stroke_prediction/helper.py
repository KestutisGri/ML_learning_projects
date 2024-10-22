import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import phik
sns.set_palette('deep')  # Set the color palette to 'pastel'


def detect_outliers(df, cols_to_check=None):
    """
    Detects outliers in the specified columns of a DataFrame using the IQR method.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check for outliers.
    cols_to_check (list, optional): A list of column names to check for outliers. If None, checks all numerical columns.

    Returns:
    None
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If cols_to_check is not provided, check all numerical columns
    if cols_to_check is None:
        cols_to_check = numerical_cols

    # Check only numerical columns for outliers
    for col in cols_to_check:
        if col in numerical_cols:
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


sns.set_palette('deep')  # Set the color palette to 'pastel'


# Your plot_data function here
def plot_data(df, plot_specs):
    """
    Plots the specified columns of a DataFrame according to the given specifications.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    plot_specs (list of tuples): A list of tuples, where each tuple contains the column name and the plot type.

    Returns:
    None
    """
    num_plots = len(plot_specs)
    num_rows = num_plots // 2 + num_plots % 2

    # Adjust the figure height based on the number of rows
    fig_height = num_rows * 2.5  # Adjust the multiplier as needed

    fig, axs = plt.subplots(num_rows, 2, figsize=(20, fig_height))

    axs = axs.ravel()  # Flatten the array to easily manage the subplots

    for i, (col, plot_type) in enumerate(plot_specs):
        if plot_type == 'hist':
            sns.histplot(df[col], kde=True, ax=axs[i])
        elif plot_type == 'count':
            ax = sns.countplot(x=col, data=df, ax=axs[i])
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
        axs[i].set_title(f'{col} {plot_type}')

    # Remove unused subplots
    if num_plots % 2 != 0:
        fig.delaxes(axs[-1])

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


def plot_multiple_confusion_matrices(y_true_pred_list, titles=None):
    """
    Plots confusion matrices for multiple models using seaborn's heatmap.

    Parameters:
    y_true_pred_list (list of tuples): A list of tuples, where each tuple contains the true labels and predicted labels for a model.
    titles (list of str, optional): List of titles for the heatmaps. Defaults to None.

    Returns:
    None
    """
    num_plots = len(y_true_pred_list)
    num_rows = num_plots // 2 + num_plots % 2
    num_cols = 2 if num_plots > 1 else 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 7))
    axes = axes.ravel()  # Flatten the axes array to easily manage the subplots

    # If the number of subplots is more than the number of plots, delete the extra subplots
    if len(axes) > num_plots:
        for ax in axes[num_plots:]:
            fig.delaxes(ax)

    for i, (y_true, y_pred) in enumerate(y_true_pred_list):
        conf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Truth')
        if titles:
            axes[i].set_title(titles[i])

    plt.tight_layout()
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


def plot_features(data, numeric_features, categorical_features, exclude_columns=[]):
    """
    Plots the specified numeric and categorical features of a DataFrame.
    The function creates a grid of subplots with 2 columns, where each subplot is a distribution plot of a feature.
    Numeric features are plotted as KDE plots, and categorical features are plotted as bar plots.
    The function also annotates the percentage values on the bars of the categorical plots.

    Parameters:
    data (pandas.DataFrame): The DataFrame to plot.
    numeric_features (list): A list of numeric feature names to plot.
    categorical_features (list): A list of categorical feature names to plot.
    exclude_columns (list, optional): A list of feature names to exclude from plotting. Defaults to an empty list.

    Returns:
    None
    """
    # Exclude specified columns
    numeric_features = [col for col in numeric_features if col not in exclude_columns]
    categorical_features = [col for col in categorical_features if col not in exclude_columns]

    # Calculate the number of rows needed for the grid
    num_rows = len(numeric_features + categorical_features) // 2 if len(
        numeric_features + categorical_features) % 2 == 0 else len(numeric_features + categorical_features) // 2 + 1

    # Create subplots with 2 columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

    # Flatten the axes array
    axes = axes.ravel()

    # Plotting numeric features
    for i, feature in enumerate(numeric_features):
        sns.kdeplot(data=data[data['stroke'] == 1], x=feature, color='red', label='Stroke', ax=axes[i])
        sns.kdeplot(data=data[data['stroke'] == 0], x=feature, color='blue', label='No Stroke', ax=axes[i])
        axes[i].set_title(f'{feature}', fontsize=14)
        axes[i].legend()

    # Plotting categorical features
    for i, feature in enumerate(categorical_features, start=len(numeric_features)):
        stroke_1 = data[data['stroke'] == 1][feature].value_counts(normalize=True)
        stroke_0 = data[data['stroke'] == 0][feature].value_counts(normalize=True)
        stroke_1.plot(kind='bar', color='red', alpha=0.5, ax=axes[i], position=0, width=0.4)
        stroke_0.plot(kind='bar', color='blue', alpha=0.5, ax=axes[i], position=1, width=0.4)
        axes[i].set_title(f'{feature}', fontsize=14)
        axes[i].legend(['Stroke', 'No Stroke'])

        # Add percentage values on bars
        for p in axes[i].patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            axes[i].annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center')

    # Remove unused subplots
    if len(numeric_features + categorical_features) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


from sklearn.metrics import roc_curve, auc


def plot_multiple_roc_curves(models, model_names, X_tests, y_tests, titles=None):
    """
    Plots ROC curves for multiple models.

    Parameters:
    models (list): List of trained models.
    model_names (list): List of names corresponding to the models.
    X_tests (list): List of test data for each model.
    y_tests (list): List of true labels for each test data.
    titles (list, optional): List of titles for each plot. Defaults to None.

    Returns:
    None
    """
    num_plots = len(models)
    num_rows = num_plots // 2 + num_plots % 2
    num_cols = 2 if num_plots > 1 else 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 7))
    axes = axes.ravel()

    if len(axes) > num_plots:
        for ax in axes[num_plots:]:
            fig.delaxes(ax)

    for i, (model, X_test, y_test, model_name) in enumerate(zip(models, X_tests, y_tests, model_names)):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, label=f"AUC ({model_name}) = {auc_score:.2f}")
        axes[i].plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')
        axes[i].set_title(titles[i] if titles else f"ROC Curve for {model_name}")
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].legend(loc=4)
        axes[i].grid()

    plt.tight_layout()
    plt.show()


from scipy.stats import chi2_contingency, chi2
import pandas as pd


def perform_chi2_test(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> None:
    """
    Performs a Chi-Square test of independence.

    Parameters:
    df (pd.DataFrame): DataFrame - the data
    col1 (str): String - the name of the first column in the data
    col2 (str): String - the name of the second column in the data
    alpha (float): Float - significance level (default is 0.05)

    Returns:
    None: The function prints the Degrees of Freedom, Critical Value, Chi-Square Statistic, and P-value of the test.
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    critical_value = chi2.ppf(1 - alpha, dof)
    print(f"Degrees of Freedom: {dof}")
    print(f"Critical Value: {round(critical_value, 3)}")
    print(f"Chi-Square Statistic: {round(chi2_stat, 3)}")
    print(f"P-value: {round(p_value, 7)}")


from sklearn.metrics import classification_report
import pandas as pd


def generate_classification_reports(models, X_train, y_train, X_val, y_val):
    """
    Function to generate classification reports for multiple models.

    Parameters:
    models (dict): A dictionary where the keys are model names and the values are sklearn model instances.
    X_train (DataFrame): The training data.
    y_train (Series): The labels for the training data.
    X_val (DataFrame): The validation data.
    y_val (Series): The labels for the validation data.

    Returns:
    final_report (DataFrame): A DataFrame containing the classification reports for each model.
    """
    classification_reports = []

    # Iterate over the models
    for model_name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_val)

        # Generate classification report
        report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=1)

        # Convert the report dictionary to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # Set the model name as the first level of the index
        report_df.index = pd.MultiIndex.from_product([[model_name], report_df.index])

        # Append the DataFrame to the list
        classification_reports.append(report_df)

    # Concatenate all the DataFrames in the list
    final_report = pd.concat(classification_reports)

    return final_report


import matplotlib.pyplot as plt
import numpy as np


def plot_classification_distribution(proba_list, title_list, figsize=(10, 5)):
    """
    Function to plot the distribution of classification probabilities.

    Parameters:
    proba_list (list of ndarray): List of arrays containing the probabilities for each class.
    title_list (list of str): List of titles for each subplot.
    figsize (tuple): Tuple representing the width and height of the figure in inches.

    Returns:
    None
    """
    num_plots = len(proba_list)
    fig, axs = plt.subplots(num_plots, figsize=figsize)

    for i in range(num_plots):
        counts, bins, patches = axs[i].hist(proba_list[i], bins=10, edgecolor='k')
        axs[i].set_title(title_list[i])
        axs[i].set_xlabel('Classification Scores')
        axs[i].set_ylabel('Frequency')

        # Add values on top of the bars
        for count, bin, patch in zip(counts, bins, patches):
            height = patch.get_height()
            axs[i].text(bin + (bins[1] - bins[0]) / 2, height, str(int(count)), fontsize=12, ha='center',
                        fontweight='bold')

    plt.tight_layout()
    plt.show()
