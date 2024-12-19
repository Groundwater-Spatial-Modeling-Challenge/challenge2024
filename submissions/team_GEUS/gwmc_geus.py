import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import json

#%% Preproc Data

# Read training, test, and prediction datasets into DataFrames
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
prediction = pd.read_csv('./data/prediction.csv')

# Extracting and Aligning Covariate Names

# Get covariate (feature) names from the test DataFrame
covariates_names = test.columns.values

# Ensure the prediction DataFrame has the same column names as the test DataFrame
prediction.columns = test.columns.values

# Handling Categorical Covariates

# Read the list of categorical covariates from a file
cat_covariates = pd.read_csv('nominal.txt')

# Converting Categorical Columns to Integer Codes

# Loop through each categorical covariate and convert to integer codes
for i in range(0, len(cat_covariates)):
    categorical_column = cat_covariates.iloc[i].values[0]  # Get the categorical column name
    
    # Convert the column in the prediction DataFrame to a categorical type
    prediction[categorical_column] = pd.Categorical(prediction[categorical_column])
    categories = prediction[categorical_column].cat.categories  # Store categories for consistency
    
    # Apply the same categories to the test and train DataFrames
    test[categorical_column] = pd.Categorical(test[categorical_column], categories=categories)
    train[categorical_column] = pd.Categorical(train[categorical_column], categories=categories)
    
    # Convert the categorical columns to integer codes
    prediction[categorical_column] = prediction[categorical_column].cat.codes
    test[categorical_column] = test[categorical_column].cat.codes
    train[categorical_column] = train[categorical_column].cat.codes

#%% Hyperparameter Tuning with GridSearchCV

# Define a grid of hyperparameters for tuning
param_grid = {
    'n_iter': [512, 1024],
    'num_threads': [2],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'num_leaves': [16, 32, 64, 128, 256],
    'max_depth': [-1, 16, 32, 64, 128],
    'colsample_bytree': [0.6, 0.8, 1],
    'subsample': [0.6, 0.8, 1],
}

# Initialize LightGBM regressor with mean squared error objective
lgb_estimator = lgb.LGBMRegressor(boosting_type='gbdt', objective='mse', random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=lgb_estimator,
    param_grid=param_grid,
    cv=3,          # 3-fold cross-validation
    verbose=2,     # Print progress
    n_jobs=48      # Use 48 parallel jobs
)

# Fit the GridSearchCV on the training data
grid_search.fit(train[covariates_names], train['target_nitrate'])

# Get the best hyperparameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Save the best hyperparameters to a file
with open('best_params.txt', 'w') as file:
    file.write(json.dumps(best_params))  # Save as JSON string

#%% Training and Cross-Validation

# Define model parameters for LightGBM training
params = {
    'task': 'train',
    'objective': 'regression',
    'boosting': 'gbdt',
    'num_leaves': 16,
    'eta': 0.005,          # Learning rate
    'metric': {'mse'},     # Mean squared error metric
    'n_iter': 1024,        # Number of iterations
    'max_depth': -1,       # No maximum depth limit
    'colsample_bytree': 0.6,
    'subsample': 0.6,
    'num_threads': 15,     # Use 15 threads for parallel processing
    'verbose': -1          # Suppress verbose output
}

# 5-fold cross-validation with fixed random state for reproducibility
kf = KFold(n_splits=5, random_state=99, shuffle=True)

# Array to store statistics: mean error, mean absolute error, RMSE, correlation
stats = np.empty((4, 5))
i = 0

for train_index, test_index in kf.split(train):
    # Split data into training and testing folds
    df_train, df_test = train.loc[train_index], train.loc[test_index]
    X_train, X_test = df_train[covariates_names], df_test[covariates_names]
    y_train, y_test = df_train['target_nitrate'], df_test['target_nitrate']
    
    # Create LightGBM dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    
    # Train the model
    ml = lgb.train(params, train_set=lgb_train)
    
    # Predict on the test fold
    predict_test = ml.predict(X_test)
    
    # Plot observed vs predicted values
    plt.figure(i + 1)
    plt.title('cv: ' + str(int(i + 1)))
    plt.plot([0, 100], [0, 100], 'r--')  # Diagonal reference line
    plt.scatter(y_test, predict_test, c='k', s=10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.tight_layout()
    
    # Calculate statistics for the current fold
    stats[0, i] = np.mean(predict_test - y_test)  # Mean error
    stats[1, i] = np.mean(np.abs(predict_test - y_test))  # Mean absolute error
    stats[2, i] = np.sqrt(np.mean(np.power((predict_test - y_test), 2)))  # RMSE
    stats[3, i] = np.corrcoef(predict_test, y_test)[0, 1]  # Correlation coefficient
    
    i += 1

# Create a DataFrame of the mean statistics across folds
stats_out = pd.DataFrame(
    index=['me', 'mae', 'rmse', 'r'],
    data=np.round(np.mean(stats, axis=1), 6)
)
print(stats_out)

#%% Final Training and Predictions

# Create a LightGBM dataset using the full training data
lgb_train = lgb.Dataset(train[covariates_names], train['target_nitrate'])

# Train the main LightGBM model with the specified parameters
ml = lgb.train(params, train_set=lgb_train)

# Update parameters to configure the model for quantile regression at the lower bound (2.5th percentile)
params.update({"metric": "quantile"})       # Change metric to quantile
params.update({"objective": "quantile"})    # Change objective to quantile regression
params["alpha"] = 0.025                     # Set alpha for the 2.5th percentile

# Train the lower-bound quantile regression model
ml_low = lgb.train(params, train_set=lgb_train)

# Update alpha for the upper bound (97.5th percentile)
params["alpha"] = 0.975

# Train the upper-bound quantile regression model
ml_high = lgb.train(params, train_set=lgb_train)

# Predict the target nitrate values and their confidence intervals for the train dataset
predict_tr = ml.predict(train[covariates_names])          # Main prediction
predict_low_tr = ml_low.predict(train[covariates_names])  # Lower-bound prediction
predict_high_tr = ml_high.predict(train[covariates_names])# Upper-bound prediction

# Predict the target nitrate values and their confidence intervals for the test dataset
predict_te = ml.predict(test[covariates_names])
predict_low_te = ml_low.predict(test[covariates_names])
predict_high_te = ml_high.predict(test[covariates_names])

# Predict the target nitrate values and their confidence intervals for the prediction dataset
predict_p = ml.predict(prediction[covariates_names])
predict_low_p = ml_low.predict(prediction[covariates_names])
predict_high_p = ml_high.predict(prediction[covariates_names])

# Create a DataFrame for train data with predicted values and confidence intervals
dataset_train = pd.DataFrame({
    'id': train.id,
    'target_nitrate': predict_tr,
    'target_nitrate_ub': predict_high_tr,  # Upper bound
    'target_nitrate_lb': predict_low_tr    # Lower bound
})

# Create a DataFrame for test data with predicted values and confidence intervals
dataset_test = pd.DataFrame({
    'id': test.id,
    'target_nitrate': predict_te,
    'target_nitrate_ub': predict_high_te,
    'target_nitrate_lb': predict_low_te
})

# Create a DataFrame for prediction data with predicted values and confidence intervals
dataset_prediction = pd.DataFrame({
    'id': prediction.id,
    'target_nitrate': predict_p,
    'target_nitrate_ub': predict_high_p,
    'target_nitrate_lb': predict_low_p
})

# Concatenate the train and test DataFrames into one DataFrame
dataset_traintest = pd.concat([dataset_train, dataset_test])

# Sort the combined DataFrame by the 'id' column to maintain order
dataset_traintest = dataset_traintest.sort_values(by='id')

# Reset the index after sorting
dataset_traintest = dataset_traintest.reset_index(drop=True)

# Save dataset_traintest to CSV
dataset_traintest.to_csv('submission_traintest.csv', index=False)

# Save dataset_prediction to CSV
dataset_prediction.to_csv('submission_prediction.csv', index=False)

   