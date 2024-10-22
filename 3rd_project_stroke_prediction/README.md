# Stroke Prediction Project

## Introduction

This project aims to predict the likelihood of a stroke using various machine-learning models. Stroke is a significant cause of mortality worldwide, and early prediction can substantially improve patient prognosis and survival rates. The project is divided into two notebooks: `eda_stroke.ipynb` and `ml_models_conclusions.ipynb`.

## Notebooks Overview

1. `eda_stroke.ipynb`: This notebook focuses on the Stroke Prediction Dataset's exploratory data analysis (EDA). It includes data cleaning, overview, visualizations, and hypothesis testing. The notebook provides a comprehensive understanding of the dataset, including the distribution of different features and their potential relationship with stroke occurrence.

2. `ml_models_conclusions.ipynb`: This notebook is dedicated to building and evaluating machine learning models for stroke prediction. It includes implementing various models, tuning hyperparameters, evaluating models, and interpreting results.

## Technical Details

The project is implemented using Python and the following libraries:

1. numpy
2. pandas
3. seaborn
4. matplotlib
5. scikit-learn
6. phik
7. scipy

## Data Sources

The data used in this project is the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.

## Methodology

The project uses the following machine-learning models:

1. XGBoost Classifier
2. Random Forest Classifier
3. CatBoost Classifier
4. LightGBM Classifier

## Results

The hyperparameter-tuned LGBM model's evaluation on the test set shows a recall score of 0.76 and a precision score of 0.1. The F1 score is 0.18. Although these results are worse than those on the validation set, the goal of increasing the recall score from 0.08 was successfully achieved.

## Results Visualization

The project's results are visualized using various plots and charts. The SHAP summary plot highlights the main features affecting the model's predictions.

## Future Work

The analysis can be improved in several ways:

1. Data Collection: Gather more data, particularly for the minority class (stroke cases), to enhance the model's performance.
2. Feature Engineering: Experiment with creating new features relevant to the prediction.
3. Model Experimentation: Try different machine learning models or ensemble methods.
4. Threshold Adjustment: Adjust the threshold for classification to improve recall.
5. Conformal Prediction: Implement conformal prediction, a technique that measures confidence in predictions.

## Installation

To run this project, you must install Python on your machine. You can then install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

## License

MIT
