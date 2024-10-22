# Travel Insurance Purchase Prediction

This project is centered around predicting potential customers who are inclined to buy travel insurance. The data used in this project is obtained from Kaggle, which provides information about nearly 2000 customers offered travel insurance in 2019.

## Objective of the Project

The main goal of this project is to build a machine learning model that can accurately predict if a customer will buy travel insurance based on their characteristics.

## Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn
- Matplotlib
- Pandas
- Numpy
- Scipy

## Implemented Models

- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

## Installation Process

1. Clone the repository
2. Install the necessary dependencies using `pip install -r requirements.txt`
3. Launch Jupyter Notebook to view the `.ipynb` file

## How to Use

To see the project in action, open the `Travel_insurance.ipynb` file in Jupyter Notebook and execute the cells.

## Outcomes

Among the models used, the Random Forest model stood out as the most effective and was chosen for further optimization and final testing. The model's performance on the test set surpassed that on the validation set in predicting customers who would purchase travel insurance.

## Potential Enhancements

- Experimentation with other models like Gradient Boosting, XGBoost, and Neural Networks could be beneficial.
- The class imbalance in the dataset could be addressed using techniques like oversampling the minority class, undersampling the majority class, or combining both (SMOTE).
- Random Search or Bayesian Optimization could be performed more comprehensively for optimal hyperparameters.
- The model's performance could be improved by incorporating external data related to the customers.
