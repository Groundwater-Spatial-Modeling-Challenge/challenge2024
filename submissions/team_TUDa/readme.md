# Team TUDa
## Author(s)

- Juan Carlos Richard-Cerda (Technische Universität Darmstadt, Institute of Applied Geosciences, Darmstadt, Germany): [https://orcid.org/0000-0001-9020-2973](https://orcid.org/0000-0001-9020-2973)
- Stephan Schulz (Technische Universität Darmstadt, Institute of Applied Geosciences, Darmstadt, Germany): [https://orcid.org/0000-0001-7060-7690](https://orcid.org/0000-0001-7060-7690)

## Feature engineering description

Feature engineering involved calculating new features from distance measurements and encoding categorical features. 
Distances from the centroid and four reference points (NW, NE, SW, SE) were transformed into (X, Y) coordinates using distance-based equations. 
Multiple solutions can exist due to symmetry, but all maintain consistent relative distances to the reference points.
Missing values were filled using Inverse Distance Weighting (IDW) interpolation. 
Additionally, categorical features were label-encoded to convert them into numerical form. 
Highly correlated features (>0.9) were removed to reduce multicollinearity. 
The final feature set was selected using SHAP values, retaining the top 25 most important features for each model (Random Forest, LightGBM, and CNN).

## Training strategy

The training strategy employed an 80/20 split of the dataset into training and testing sets. 
Cross-validation was utilized for all models to ensure robust evaluation and to optimize ensemble weights. 
Hyperparameter tuning for the models was conducted using Optuna, incorporating cross-validation to minimize error. 
To enhance model performance and reduce dimensionality, SHAP analysis was employed to select the top 25 most important features for each model.

## Model description

The model ensemble consists of three individual models: a Random Forest Regressor, a LightGBM Regressor, and a 1D Convolutional Neural Network (CNN).
Each model was trained separately on the top 25 features, which were selected based on SHAP values obtained from a previous model run using all non-highly correlated features.
An ensemble of the three models was created using optimized weights for each model's predictions.
The final ensemble output is a weighted sum of predictions from each model, with weights optimized using a constrained minimization approach to reduce the overall error.
Uncertainty quantification was performed using model variance and standard error calculations to create 95% confidence intervals for predictions.

## Software

The project was developed in Python. The key libraries used include:
- `pandas` and `numpy` for data processing.
- `scikit-learn` for model training, preprocessing, and evaluation.
- `LightGBM` for the gradient boosting model.
- `tensorflow` for the CNN model.
- `shap` for SHAP values calculation to analyze feature importance.
- `joblib` for model persistence.
- `scipy` for optimizing the ensemble weights.
- `optuna` for hyperparameter optimization.

## Estimation of effort

| Development time (hrs) | Calibration time (s) |  Prediction time (s) | 
|------------------------|----------------------|----------------------|
| ~  50                 | 1354   | 80   |


## Additional information


*** 1. Data Preparation ***
1. **Load Data**: Import the CSV dataset `train.csv`.
2. **Set Random Seeds**: Set random seeds for reproducibility.
3. **Compute Coordinates (X, Y): Calculate (X, Y) using distances from reference points (NW, NE, SW, SE). 
	 These distances represent how far the target is from each reference point. The (X, Y) is found by solving distance-based equations, 
	 where multiple solutions can exist due to symmetry, but all maintain consistent relative distances to the reference points.
4. **Drop Original Distance Features**: Remove the original distance-based features.
5. **Encode Categorical Features**: Convert categorical variables into numerical format using `LabelEncoder`.
6. **Prepare Feature (X) and Target (y) Sets**: Separate `X` (features) from `y` (target) and drop irrelevant columns.

*** 2. Feature Selection & Engineering ***
1. **Compute Correlation**: Calculate correlations among features and remove highly correlated features (correlation > 0.9).
2. **Standardize Features**: Normalize feature values using `StandardScaler`.

*** 3. Model Training (Initial Models) ***
1. **Split Data**: Split data into train (80%) and test (20%) sets.
2. **Training**: Train a Random Forest Regressor, LightGBM Regressor, and CNN with optimized hyperparameters using Optuna.

*** 4. Feature Importance using SHAP ***
1. **Compute SHAP Values**:
   - Random Forest SHAP values are calculated using `shap.Explainer`.
   - LightGBM SHAP values are calculated using `shap.Explainer`.
   - CNN SHAP values are calculated using `shap.KernelExplainer` with background and sample points.
2. **Select Top 25 Features**:
   - Extract the 25 most important features for each model (RF, LGB, CNN) based on the SHAP values.
   - If the CNN has fewer than 25 important features, additional features are included to ensure a minimum of 25 features.

*** 5. Feature Reduction and Re-train Models with Reduced Feature Sets***
1. **Reduce Feature Set**:
   - Extract only the top 25 important features for each model (Random Forest, LightGBM, and CNN).
2. **Re-train models**: Train the models on the top 25 respective features.

*** 6. Model Predictions ***
1. **Make Predictions**:
   - Random Forest makes predictions using the top 25 features.
   - LightGBM makes predictions using the top 25 features.
   - CNN makes predictions using the top 25 features.

*** 8. Ensemble Model Creation ***
1. **Weight Optimization**:
   - Define an objective function to minimize the Mean Absolute Error (MAE) of the ensemble.
   - Use `scipy.optimize.minimize` to find the optimal weights `(w_rf, w_cnn, w_lgb)` for each model.
   - Constrain the sum of the weights to be 1 (`w_rf + w_cnn + w_lgb = 1`).

2. **Make Ensemble Predictions**:
   - Use the optimized weights to make predictions:
     `y_pred_ensemble = w_rf * y_pred_rf + w_cnn * y_pred_cnn +  w_lgb * y_pred_lgbm'

*** 9. Prediction Intervals (Uncertainty) ***
   - Combine model predictions for each test point.
   - Calculate the 2.5th and 97.5th percentiles for each prediction, giving a 95% confidence interval for each prediction.

*** 10. Model Evaluation ***
1. **Evaluate Ensemble**:
   - Calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) of the final ensemble predictions.
2. **Save Performance Metrics**:
   - Save the MAE, MSE, RMSE, and final ensemble weights for Random Forest, CNN, and LightGBM.

*** 11. Use Model for Prediction ***
	- Use IDW for fillig missing data from nearest neighbours.
	- Run model and create predictions.
