## Football Match Prediction Project

This project aims to develop a predictive model that can outperform bookmakers' odds in predicting the outcome of a football match, specifically the home team's victory. The project also aims to construct a model that can accurately predict whether a football match will end with 'Under' or 'Over' 2.5 goals.

The project is divided into several phases, each represented by a different file. It's important to note that these files should be run in the order they are listed, as each file generates new CSV files used in the subsequent files.

### 1. `data_cleaning_feature_engineering.ipynb`

This file is responsible for cleaning the initial data and engineering the features. It includes:

- Importing necessary libraries and establishing a connection to the SQLite database.
- Extracting essential data from various tables.
- Cleaning the data by addressing null values, duplicates, and data types.
- Generating several new features is expected to enhance the predictive power of our models.
- Saving the cleaned and enriched data for further analysis in a CSV file.

### 2. `eda_hypothesis.ipynb`

This file is responsible for exploratory data analysis and hypothesis formulation. It includes:

- Understanding the data and identifying potential features for our predictive models.
- Visualizing the data to gain insights and formulate hypotheses.
- The data used in this file is loaded from the CSV file generated in the previous step.

### 3. `home_team_win_model.ipynb`

This file is responsible for constructing and evaluating various machine-learning models to predict the home team's victory. It includes:

- Training models on the cleaned and enriched data.
- Evaluating models using appropriate metrics.
- Selecting the model with the best performance for further optimization.
- Testing the model's predictions against the bookmakers' odds for the 2015/2016 season.
- The data used in this file is loaded from the CSV file generated in the first step.

### 4. `goals_models.ipynb`

This file predicts whether a football match will end with 'Under' or 'Over' 2.5 goals. It includes:

- Employing several machine learning algorithms to train our model.
- Evaluating the model's performance and selecting the best model for further optimization.
- Evaluating our model using metrics like ROC AUC score, ROC curve, and confusion matrix.
- The data used in this file is loaded from the CSV file generated in the first step.

