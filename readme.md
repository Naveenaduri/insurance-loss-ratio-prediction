# XGBoost Insurance Loss Prediction

This Colab notebook demonstrates building and evaluating machine learning models for predicting insurance loss amounts using XGBoost and K-Nearest Neighbors (KNN) regressors.

## Data Preparation Steps

1. **Import Libraries**
    - Pandas
    - Seaborn
    - NumPy
    - Matplotlib

2. **Install XGBoost**
    ```bash
    !pip3 install xgboost
    ```

3. **Load Data**
    - Read training data from CSV (`./train/training_data.csv`)

4. **Explore Data**
    - Check data shape and missing values
    - Analyze unique values in each column
    - Create separate dataframes for numerical and categorical features

5. **Feature Engineering**
    - Define features used for modeling
    - Convert categorical features to one-hot encoded format
    - Handle missing values (e.g., strip leading/trailing spaces)
    - Calculate mean and sum of numerical and categorical features for each portfolio

## Modeling

1. **Split Data**
    - Separate features and target variable (`Loss_Amount`) from training data

2. **Train XGBoost Model**
    - Define XGBoost Regressor with hyperparameters (objective, learning rate, etc.)
    - Fit the model on normalized training data (`np.nan_to_num(X_train.to_numpy())`)

3. **Evaluate XGBoost Model**
    - Predict loss amounts on training data
    - Calculate Mean Absolute Error (MAE) between actual and predicted values

4. **Train KNN Model**
    - Create a KNN Regressor with a specified number of neighbors (e.g., 5)
    - Fit the model on normalized training data

5. **Evaluate KNN Model**
    - Predict loss amounts on training data
    - Calculate MAE between actual and predicted values

## Testing

1. **Load Test Portfolios**
    - Read multiple test portfolio files from a directory (`./test/testing_portfolios`)
    - Process each portfolio:
        - Select relevant features
        - Convert categorical features to one-hot encoded format
        - Impute missing values
        - Reindex features to match training data

2. **Prepare Testing Dataframe**
    - Create a new dataframe with features for all test portfolios
    - Add an "ID" column for portfolio identification
    - Calculate mean and sum of features for each test portfolio (similar to training data)

3. **Generate Test Results**
    - Use the trained KNN regressor to predict loss amounts for each test portfolio
    - Calculate loss ratio (predicted loss / annual premium)
    - Calculate log of loss ratio
    - Create a results dataframe with portfolio ID and log loss ratio
    - Save predictions to a CSV file ("predictions_xgb.csv")

## Note

- This notebook focuses on KNN model evaluation for demonstration purposes.
- Uncomment the `generate_test_results` line with the XGBoost regressor to generate predictions with XGBoost and create a separate CSV file.
- Data files are not included due to a sharing agreement with the professor.


## Enhancements

Consider further enhancing the notebook by:

- Experimenting with different XGBoost hyperparameters
- Adding functionalities for grid search or hyperparameter tuning
- Including additional evaluation metrics (e.g., MAPE)
- Implementing feature selection techniques
