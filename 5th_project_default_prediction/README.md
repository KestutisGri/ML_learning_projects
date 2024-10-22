# Home Credit Default Risk Prediction

[Home Credit Default Risk Competition on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

## Data
This project involves analyzing the Home Credit Default Risk dataset from Kaggle. The dataset includes various features related to credit applications, such as application data, historical credit card balance data, information on previous installment payments, point of sale cash balance data, data on previous loan applications, and credit bureau data and balance information.

## Main Goals
The primary goal is to build a predictive model with a ROC-AUC score of at least 0.77. This metric was chosen because it was used to evaluate the best model in the Kaggle competition. The model aims to provide accurate probabilities for credit default, which is crucial for making informed lending decisions. Additionally, we will use the Brier score as a metric to ensure the accuracy of the predicted probabilities.

## Methodology
The methodology involves several steps:
1. **Exploratory Data Analysis** (EDA) involves understanding the data distribution and identifying  patterns.
2. **Feature Engineering**: Creating new features that may improve the model's predictive power.
3. **Model Training**: Training various machine learning models and evaluating their performance using ROC-AUC.
4. **Model Calibration**: Using techniques like Platt Scaling or Isotonic Regression to ensure the model provides accurate probability estimates.
5. **Evaluation**: Evaluating the model using the Brier Score and other relevant metrics.
6. **Deployment**: The model is deployed at Google Cloud, https://default-435318.lm.r.appspot.com/predict. You have to enter SK_ID_CURR from 1 to 30752, and the outcome will be probabilities of default and values of the most important features.

## Feature Engineering
Feature engineering was a crucial part of this project. New features such as `EXT_SOURCE_MEAN` were created, which was very important for the model's performance. Other features included ratios and differences between income and expenses or credit and cumulative sums for payment behaviors.

## Model Selection and Tuning
Different models, including LightGBM, XGBoost, and CatBoost, were tried. Various pipelines were tested, and hyperparameters were tuned to optimize performance. The models were also calibrated to ensure they provided reliable probability estimates.

## Insights and Interpretations
SHAP (SHapley Additive exPlanations) values were used to understand the most important features contributing to the model's predictions. This method provided insights into each feature's impact on the model's output. For instance, features like `EXT_SOURCE_1` and `EXT_SOURCE_3` were highly correlated with the target variable. 

## Results
The final model achieved a ROC-AUC score of 0.782 and a Brier score of 0.066, surpassing the target of 0.77.

## Conclusion
By following a structured approach, the project successfully developed a robust predictive model that meets the target ROC-AUC score and provides reliable probability estimates for credit default risk.
