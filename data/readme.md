# Data

This folder contains all the data for the challenge. There are 3 .csv files:

- train.csv: this file contains a ID column (id), the column containing the target variable (target_nitrate) and all feature columns (feat_...). This file contains all data that you can use to train or calibrate your model.
- test.csv: this file contains a ID column (id) and all feature columns (feat_...). The target column is missing of course. This file contains all features of the samples in the test split. Use this file to make predictions on the test split. 
- prediction.csv: this file contains a ID column (cell_id) and all feature columns (feat_...). The target column is missing of course. This file contains all features belonging to a regularly sampled grid. Use this file to make predictions for. We will use this file in order to generate spatially continuous raster maps for a visual comparison. 

If you have any questions about the data, please post them in the appropriate GitHub Discussion forum. This 
ensures that all participants will have the same information available.
