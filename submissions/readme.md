# Model results

This folder contains the folders with the model results from the different participating teams. To submit your model 
results, please: 

1. [fork](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/fork) this repository;
2. Copy the 'team_example' folder and rename (e.g., 'team_XX');
3. Change the files in the folder;
4. Create a [Pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to the original repository to submit your model.

If you need any help with this submission, please post in the GitHub Discussion Forum.

> [!NOTE]
> Please do not change the names of the files contained in the example folder, only change the data within 
the files:

- readme.md: Here you can provide information about yourself, the feature engineering, the model etc.
- submission_traintest.csv: This table should contain the id column (id), a column with your predictions for the nitrate concentrations (target_nitrate) and optionally two more columns for the prediction intervals (target_nitrate_ub, target_nitrate_lb) 
- submission_prediction.csv: This table should contain the cell id column (cell_id), a column with your predictions for the nitrate concentrations (target_nitrate) and optionally two more columns for the prediction intervals (target_nitrate_ub, target_nitrate_lb)




