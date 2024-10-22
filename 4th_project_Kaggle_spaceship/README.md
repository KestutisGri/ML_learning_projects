## Data
The dataset used in this project is the [Spaceship Titanic Dataset](https://www.kaggle.com/competitions/spaceship-titanic) from Kaggle. It includes various features related to the passengers aboard the spaceship, such as age, cabin, spending, home planet, destination, and whether they were transported to another dimension.

### Methodology
The project took a structured approach to tackling the predictive modeling challenge. Initially, a thorough exploratory data analysis (EDA) was conducted to understand the dataset's characteristics, identify missing values, and detect outliers. This phase also involved visualizing the distributions of various features and their relationships with the target variable, 'Transported.'

### Feature Engineering
Based on insights from EDA, several new features were engineered to enhance model performance. This included:
- **Encoding Categorical Variables**: Transforming categorical variables into a format that could be provided to ML models to improve prediction accuracy.
- **Creating Interaction Features**: Generating new features representing interactions between two or more variables, hypothesizing that these interactions might significantly impact the target variable.
- **Log Transformation**: Applying log transformation to skewed numerical features to normalize their distribution.
- **Handling Missing Values**: Strategically imputing missing values based on insights from the data, such as filling missing values in service-related columns with 0 for passengers in cryosleep.

### Model Selection and Tuning
Various machine learning models, including XGBoost, Random Forest, CatBoost, and LightGBM classifiers, were evaluated. Each model was carefully selected for its ability to handle the dataset's complexity and performance in similar tasks. Hyperparameter tuning was performed using the Optuna library to find the optimal settings for each model. The models were evaluated based on accuracy, recall, precision, and F1 score, focusing on achieving high accuracy to surpass the Kaggle benchmark.

### Insights and Interpretation
SHAP (Shapley Additive exPlanations) values were used to interpret the models, providing insights into how each feature influences the prediction. This analysis revealed the most important features contributing to a passenger's likelihood of being transported to another dimension, offering a deeper understanding of the model's decision-making process.

### Results
The primary goal of the Titanic Spaceship project was to reach an accuracy of 79+ on the Kaggle test dataset. The project successfully surpassed this goal, achieving an accuracy of 81.201%. This accomplishment demonstrates the effectiveness of the chosen machine learning models and highlights the importance of thorough exploratory data analysis, feature engineering, and hyperparameter tuning in achieving high predictive accuracy. The detailed evaluation metrics further provided insights into the models' performance, showcasing their strengths and areas for improvement.
