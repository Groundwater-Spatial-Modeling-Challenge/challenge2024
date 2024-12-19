# Team Example
## Author(s)

- Julian Koch (GEUS): https://orcid.org/0000-0002-7732-3436

## Feature engineering description

There were some inconsistencies in the covariate names between prediction and train/test datasets. Prediction covariate names were overwritten to ensure consistency.

Categorical Encoding:

A list of categorical covariates was loaded from nominal.txt.
Each categorical covariate was converted to integer codes to make them compatible with the model.
Consistent category mappings were applied across the train, test, and prediction datasets to maintain uniform encoding.

## Training strategy

Validation Strategy
To assess model performance, a 5-fold cross-validation (CV) strategy was used. The KFold method from sklearn.model_selection was employed

Hyperparameter Optimization
Hyperparameter tuning was performed using Grid Search Cross-Validation with the GridSearchCV function from sklearn.model_selection. The search was conducted over a predefined grid of hyperparameters for the LightGBM model.

## Model description

LightGBM Regression Model
The primary model used in this project is a LightGBM (Light Gradient Boosting Machine) regression model. LightGBM is a gradient boosting.

Loss Function:

Main Model: Uses the Mean Squared Error (MSE) loss function for central prediction.
Quantile Models: Use the Quantile Loss function to estimate the 2.5th and 97.5th percentiles for uncertainty quantification.

Hyperparameters:
The final model was configured with hyperparameters optimized through Grid Search Cross-Validation. Notable hyperparameters include:

num_leaves: 16 (controls the complexity of the trees)
learning_rate: 0.005 (step size for each boosting iteration)
n_iter: 1024 (number of boosting iterations)
max_depth: -1 (unrestricted tree depth)
colsample_bytree: 0.6 (fraction of features used for each tree)
subsample: 0.6 (fraction of data used for each boosting iteration)
num_threads: 15 (parallel processing for faster training)

Final Training
Using the best-found hyperparameters, the final models were trained on the entire training dataset.

Uncertainty Quantification:
To provide prediction intervals, two additional LightGBM models were trained:

Lower-Bound Model: Predicts the 2.5th percentile (lower confidence bound).
Upper-Bound Model: Predicts the 97.5th percentile (upper confidence bound).

## Software

Python 3.9
Regressoin model applied: https://lightgbm.readthedocs.io/en/stable/

## Estimation of effort

| Development time (hrs) | Calibration time (s) |  Prediction time (s) | 
|------------------------|----------------------|----------------------|
| ~  4                 | 3600   | 60   |


