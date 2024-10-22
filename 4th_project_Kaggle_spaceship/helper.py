import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sns.set_palette('deep')


def detect_outliers_percentile(df, cols_to_check=None, lower_percentile=1, upper_percentile=99):
    """
    Detects outliers in the specified columns of a DataFrame using the percentile method.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check for outliers.
    cols_to_check (list, optional): A list of column names to check for outliers. If None, checks all numerical columns.
    lower_percentile (int, optional): The lower percentile to use for outlier detection. Defaults to 1.
    upper_percentile (int, optional): The upper percentile to use for outlier detection. Defaults to 99.

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
            lower_limit = np.percentile(df[col], lower_percentile)
            upper_limit = np.percentile(df[col], upper_percentile)

            outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
            sample_size = df[col].notna().sum()

            print(f"Number of outliers in {col}: {len(outliers)}")
            print(f"Sample size for {col}: {sample_size}")


def plot_outliers(df, cols_to_check, lower_percentile=1, upper_percentile=99):
    """
    Creates scatter plots for the specified columns of a DataFrame to visualize outliers.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    cols_to_check (list): A list of column names to plot.
    lower_percentile (int, optional): The lower percentile to use for outlier detection. Defaults to 1.
    upper_percentile (int, optional): The upper percentile to use for outlier detection. Defaults to 99.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    for i, col in enumerate(cols_to_check):
        plt.subplot(1, len(cols_to_check), i + 1)

        # Calculate lower and upper percentiles
        lower_limit = np.percentile(df[col], lower_percentile)
        upper_limit = np.percentile(df[col], upper_percentile)

        # Create a scatter plot with transparency
        plt.scatter(range(df.shape[0]), df[col].values, alpha=0.1)
        plt.plot([0, df.shape[0]], [lower_limit, lower_limit], color='r')
        plt.plot([0, df.shape[0]], [upper_limit, upper_limit], color='r')
        plt.title(f'Scatter plot of {col}')

    plt.tight_layout()
    plt.show()


def plot_phik(df, columns_to_drop=None, figsize=(10, 10)):
    """
    Computes the Phi-K correlation matrix for a DataFrame and plots it as a triangular heatmap.

    Parameters:
    df (pandas.DataFrame): The DataFrame to compute the Phi-K correlation matrix for.
    columns_to_drop (list, optional): The list of column names to drop from the DataFrame. Defaults to None.
    figsize (tuple, optional): The size of the figure in inches. Defaults to (10, 10).

    Returns:
    None
    """
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    phi_k_matrix = df.phik_matrix()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(phi_k_matrix, dtype=bool))

    plt.figure(figsize=figsize)
    # Use fmt='.2f' to display 2 numbers after the decimal
    sns.heatmap(phi_k_matrix, annot=True, fmt='.2f', mask=mask)
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


import numpy as np


def plot_categorical_distribution(data, variables, target):
    sns.set_palette('deep')
    data = data.copy()
    for var in variables:
        data[var] = data[var].fillna('NaN').astype(str)

    nrows = len(variables) // 2
    if len(variables) % 2: nrows += 1

    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(25, 8 * nrows))
    axs = axs.flatten() if nrows > 1 else [axs]

    for ax, column in zip(axs, variables):
        total_counts = data[column].value_counts().sort_index()
        true_counts = data[data[target] == True][column].value_counts().sort_index()
        false_counts = total_counts.subtract(true_counts, fill_value=0)

        true_percentage = (true_counts / total_counts) * 100
        false_percentage = (false_counts / total_counts) * 100

        bars1 = ax.bar(true_counts.index, true_counts, label='True', alpha=0.8)
        bars2 = ax.bar(false_counts.index, false_counts, bottom=true_counts, label='False', alpha=0.8)

        for bar, percentage in zip(bars2, false_percentage):
            height = bar.get_height() + bar.get_y()
            ax.text(bar.get_x() + bar.get_width() / 2., height - (height * 0.05), f'{percentage:.1f}%', ha='center',
                    va='top', color='black', fontsize=11)
        for bar, percentage in zip(bars1, true_percentage):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height * 0.5, f'{percentage:.1f}%', ha='center', va='center',
                    color='black', fontsize=11)

        for index, value in total_counts.items():
            ax.text(index, true_counts.get(index, 0) + false_counts.get(index, 0) + 3, str(value), ha='center')

        ax.set_xlabel(column)
        ax.set_ylabel('Counts')
        ax.set_title(f'Counts and Percentage of {target} for {column}')
        ax.legend()

    if len(variables) % 2 != 0:
        fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.show()


def plot_age_distribution(df, column):
    """
    Plots Kernel Density Estimation (KDE) plots for the specified column in a DataFrame,
    separating the data based on the 'Transported' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    column (str): The column name to plot.

    Returns:
    None
    """
    # Create two subsets of the data for transported and not transported passengers
    transported = df[df['Transported'] == True]
    not_transported = df[df['Transported'] == False]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Create a KDE plot for the specified column for transported passengers
    sns.kdeplot(transported[column], color='blue', ax=ax, label='Transported')

    # Create a KDE plot for the specified column for not transported passengers
    sns.kdeplot(not_transported[column], color='red', ax=ax, label='Not Transported')

    # Set the plot title and labels
    ax.set_title(f'Distribution of {column} for Transported and Not Transported Passengers')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')

    # Limit the x-axis to show only values greater than or equal to 0 and less than or equal to 79
    ax.set_xlim([0, 79])

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()


def plot_age_category_distribution(df, age_category_column, target_column):
    sns.set_palette('deep')

    """
    Plots a count plot for the specified age category column in a DataFrame,
    separating the data based on the target column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    age_category_column (str): The age category column name to plot.
    target_column (str): The target column name to separate the data.

    Returns:
    None
    """
    # Create a count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=age_category_column, hue=target_column, data=df)

    # Set the plot title and labels
    plt.title(f'Count of {target_column} for Each {age_category_column}')
    plt.xlabel(age_category_column)
    plt.ylabel('Count')

    # Display the plot
    plt.show()


def plot_histograms(df, columns, use_log=False):
    """
    Plots histograms for the specified columns of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    columns (list): A list of column names to plot.
    use_log (bool): If True, apply a logarithmic transformation to the data.

    Returns:
    None
    """
    # Calculate the number of rows needed for the subplots
    n_rows = int(np.ceil(len(columns) / 3))

    # Create a figure with subplots, 3 plots per line
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    # Flatten the axes array
    axs = axs.flatten()

    # Iterate over each service column
    for i, column in enumerate(columns):
        # Apply log transformation if use_log is True
        data = np.log1p(df[column]) if use_log else df[column]

        # Plot the column using Seaborn
        sns.histplot(data, ax=axs[i], kde=True, color='blue')

        # Set the title and labels
        axs[i].set_title(f'Distribution of {column}')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Frequency')

    # Remove unused subplots
    for i in range(len(columns), len(axs)):
        fig.delaxes(axs[i])

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_kde(df, column, use_log=False):
    """
    Plots Kernel Density Estimation (KDE) plots for the specified column in a DataFrame,
    separating the data based on the 'Transported' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    column (str): The column name to plot.
    use_log (bool): If True, apply a logarithmic transformation to the data.

    Returns:
    None
    """
    # Apply log transformation if use_log is True
    if use_log:
        df = df.copy()
        df[column] = np.log1p(df[column])

    # Create a KDE plot for the specified column of transported passengers
    sns.kdeplot(df[df['Transported'] == 1][column], label='Transported', fill=False)

    # Create a KDE plot for the specified column of not transported passengers
    sns.kdeplot(df[df['Transported'] == 0][column], label='Not Transported', fill=False)

    # Set the plot title and labels
    plt.title(f'KDE Plot of {column} for Transported and Not Transported Passengers')
    plt.xlabel(column)
    plt.ylabel('Density')

    # Add a legend
    plt.legend()

    # Limit the x-axis to non-negative values
    plt.xlim(0, )

    # Display the plot
    plt.show()


def plot_count(df, x_col, hue_col, title='Count Plot', figsize=(10, 6)):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x=x_col, hue=hue_col)
    plt.title(title)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # Only annotate bars with a height greater than zero
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center',
                        va='bottom',  # Adjust to 'bottom' to ensure visibility
                        xytext=(0, 10),  # Adjust this value to move the annotation up
                        textcoords='offset points')
    plt.show()


def generate_predictions(model, X_val, y_val):
    """
    Generates a DataFrame containing the validation features, actual labels, predicted labels,
    and probabilities of each class from a given model.

    Parameters:
    - model: The trained model used for predictions. Must have predict and predict_proba methods.
    - X_val (pd.DataFrame): The validation features DataFrame.
    - y_val (pd.Series): The actual labels Series.

    Returns:
    - pd.DataFrame: A DataFrame containing the original validation features, actual labels,
      predicted labels, and the probabilities for each class.
    """
    # Reset the index of both X_val and y_val to ensure alignment
    X_val_reset = X_val.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)

    # Predict the classes and probabilities
    y_val_pred = model.predict(X_val_reset)
    probabilities = model.predict_proba(X_val_reset)

    # Convert predictions and probabilities into pandas Series with the correct index
    y_val_series = pd.Series(y_val_reset, name='Actual', index=X_val_reset.index)
    y_val_pred_series = pd.Series(y_val_pred, name='Predicted', index=X_val_reset.index)
    prob_not_transported_series = pd.Series(probabilities[:, 0], name='Prob_Not_Transported', index=X_val_reset.index)
    prob_transported_series = pd.Series(probabilities[:, 1], name='Prob_Transported', index=X_val_reset.index)

    # Concatenate into a single DataFrame
    validation_full_df = pd.concat(
        [X_val_reset, y_val_series, y_val_pred_series, prob_not_transported_series, prob_transported_series], axis=1)

    return validation_full_df


import matplotlib.pyplot as plt
import seaborn as sns


def plot_classification_probabilities(validation_df, figsize=(20, 10)):
    """
    Plots the distribution of predicted probabilities for each classification scenario.

    Parameters:
    - validation_df: DataFrame containing the actual and predicted classifications,
      along with the probabilities of being classified as not transported and transported.
    - figsize: Tuple specifying the figure size.

    Returns:
    None. Displays a 2x2 grid of histograms showing the distribution of probabilities
    for each actual vs. predicted classification scenario.
    """
    # Define the figure and axes for subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Titles and colors for each subplot
    titles = ['Actual 0, Predicted 0', 'Actual 1, Predicted 1', 'Actual 0, Predicted 1', 'Actual 1, Predicted 0']
    colors = ['green', 'blue', 'red', 'orange']
    data_filters = [
        (validation_df['Actual'] == 0) & (validation_df['Predicted'] == 0),
        (validation_df['Actual'] == 1) & (validation_df['Predicted'] == 1),
        (validation_df['Actual'] == 0) & (validation_df['Predicted'] == 1),
        (validation_df['Actual'] == 1) & (validation_df['Predicted'] == 0)
    ]
    prob_columns = ['Prob_Not_Transported', 'Prob_Transported', 'Prob_Transported', 'Prob_Not_Transported']

    # Loop through each scenario to plot
    for i, (data_filter, title, color, prob_column) in enumerate(zip(data_filters, titles, colors, prob_columns)):
        ax = axes[i // 2, i % 2]  # Determine subplot position
        sns.histplot(validation_df[data_filter][prob_column], kde=True, color=color, bins=20, ax=ax)
        ax.set_title(title)
        # Adjusted x-axis label based on the prediction
        if "Predicted 0" in title:
            ax.set_xlabel('Probability Not Transported')
        else:
            ax.set_xlabel('Probability Transported')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_prediction_distributions(validation_df, features, feature_types):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter correctly and wrongly predicted samples
    correct_predictions = validation_df[validation_df['Actual'] == validation_df['Predicted']]
    wrong_predictions = validation_df[validation_df['Actual'] != validation_df['Predicted']]

    # Adjust the subplot grid for the additional features
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 5 * len(features)))
    fig.suptitle('Distribution of Correctly and Wrongly Predicted Samples by Feature')

    for i, (feature, f_type) in enumerate(zip(features, feature_types)):
        if f_type == 'categorical':
            # Calculate the order for correct predictions
            order_correct = correct_predictions[feature].value_counts().index
            # Correct predictions
            sns.countplot(x=feature, data=correct_predictions, ax=axes[i, 0], order=order_correct)
            axes[i, 0].set_title(f'Correct Predictions - {feature}')
            axes[i, 0].set_ylabel('Count')

            # Calculate the order for wrong predictions
            order_wrong = wrong_predictions[feature].value_counts().index
            # Wrong predictions
            sns.countplot(x=feature, data=wrong_predictions, ax=axes[i, 1], order=order_wrong)
            axes[i, 1].set_title(f'Wrong Predictions - {feature}')
            axes[i, 1].set_ylabel('Count')
        else:  # Numerical features
            # Correct predictions
            sns.histplot(correct_predictions[feature], kde=True, color='green', ax=axes[i, 0])
            axes[i, 0].set_title(f'Correct Predictions - {feature}')
            axes[i, 0].set_ylabel('Frequency')

            # Wrong predictions
            sns.histplot(wrong_predictions[feature], kde=True, color='red', ax=axes[i, 1])
            axes[i, 1].set_title(f'Wrong Predictions - {feature}')
            axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
