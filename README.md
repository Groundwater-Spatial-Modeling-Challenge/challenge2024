# The Groundwater Spatial Modeling Challenge 2024

This challenge is strongly inspired by the last years [**Groundwater Time Series Modeling Challenge**](https://github.com/gwmodeling/challenge) organized by R.A. Collenteur (Eawag) et al. and can be considered as an extension to the spatial domain.

This repository contains all the information and materials for the Groundwater Spatial Modeling Challenge, [as announced at the 2024 EGU General Assembly](https://meetingorganizer.copernicus.org/EGU24/EGU24-10386.html). We would like to invite every interested scientist or modeler to participate in this contest and submit the results of their best-performing models. In this way, we not only want to learn from each other, but also bring shared experience and creativity to this still very small community.

**Organisers:** E. Haaf (Chalmers), T. Liesch & M. Ohmer (KIT), M. Nölscher (BGR)

## Background & Objectives

Spatially continuous information (aka. maps) about groundwater related parameters is a crucial pre-condition for many tasks in water management and ecological questions. However, in hydrogeology, we often lack such reliable datasets due to the fact that observed data comes almost always from observation wells which reflect only a very narrow area around it. Based on such observed data, spatially continuous datasets can be then generated using deterministic methods like Voronoi-polygons or Inverse-Distance-Weighting or geostatistical methods like the Krigging-family for inter- and extrapolating over space. In the recent two decades machine learning approaches have been increasingly studied and applied for this task. Compared to time series modeling of groundwater levels, there has not yet been a comparable breakthrough in mapping/regionalizing groundwater related parameters. Whith this challenge, we want to

-   put more focus on this matter of research,
-   grow the community,
-   increase discussion,
-   showcase the diversity of approaches and evaluate their capabilities and
-   learn from each others experience and creativity.

In order to get a tiny bit closer to these goals, this challenge is about modeling nitrate concentrations (including the prediction intervals) in shallow aquifers in southwest Germany using a broad range of geophysical predictor variables aka. features. Any model is welcome. It is not restricted to machine learning models.

In the following, all necessary information on participating in the challenge is explained.

## Inputa data (features and target variable)

We split the overall dataset with approx. 1800 samples/locations into a training and a test set using a proportion of 80% for training. This dataset has contains 3 types of columns:

-   one id column: this column is only for matching the the predictions with the solutions later on
-   one target column containing the nitrate concentrations: this column name has the prefix 'target\_'
-   many feature columns containing the explanatory variables/predictors: these columns have the prefix 'feature\_'

> \[!TIP\] 
> Please find all meta data about all features in this [table](https://groundwater-spatial-modeling-challenge.github.io/challenge2024/features.html) Please find all data and descriptions in the [data folder](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/tree/main/data).

> \[!WARNING\]
>
> <p xmlns:cc="http://creativecommons.org/ns#">
>
> The data is licensed under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-ND 4.0<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"/><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"/><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"/><img src="https://mirrors.creativecommons.org/presskit/icons/nd.svg?ref=chooser-v1" style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"/></a>
>
> </p>

As we do not provide the coordinates of the samples/locations, it is not possible or allowed to add other datasets as features. But it is of course allowed to only use some of the features or calculate new features from the provided ones. Any kind of feature engineering (e.g. encoding of nominal features) is allowed. It is not permitted to use the nitrate concentration as feature itself.

## Modeling rules

-   Anyone interested can participate as single person or as team
-   Participants may use any type of model
-   The target variable namely the nitrate concentrations must not be used as predictor/feature.
-   The modeling workflow must be reproducible, preferably through the use of scripts, but otherwise described in enough detail to reproduce the results. This requires freely available software, preferably open source
-   Supplementary model data must be described in sufficient detail and submitted with model outputs.
-   Submission of model results are done via a github pull request

## Model outputs and deliverables

The model is expected to compute:

-   The prediction of the nitrate concentrations for the location ids in the submission files in the 'team_example' folder.
-   Optional: if the method/model allows the calculation of prediction intervals, the 95% prediction interval of the nitrate concentration should be added as separate columns for each location id as shown in the example file.

Forms that can be used to submit the results are provided in the [submissions folder](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/tree/main/submissions). There you can also find more details on what to submit.

## Model evaluation

The models will be evaluated using several performance metrics, computed for both the training and the test split. The data solutions for the test split are not public yet and will be released after the deadline date.

## Deadline

> \[!WARNING\] 
> The extended deadline for the challenge is now **15/07/2025 24:00 CET.** Please make sure to submit before this date. We plan to share the results of this challenge at the EGU General Assembly 2025.

## Participation & Submission

If you intend to participate, [please open a GitHub Issue for your team](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/issues), such that we can track the participating teams.

Participants can submit their model results as a Pull Request to this Repository, adding a folder with their results in the 'submissions' folder. The model results must be submitted in a way that they are reproducible, either through the use of scripts (preferred) or detailed description of the modeling process. See the [submissions folder](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/tree/main/submissions) for a more detailed description on how and what to submit.

After the challenge we intend to write an article to submit to a peer-reviewed journal with all the organisers and participants.

## Questions/ Comments ?

To make sure everyone has access to the same information we ask you to put any questions that are of general interest to all participants in the [GitHub Discussion Forum](https://github.com/Groundwater-Spatial-Modeling-Challenge/challenge2024/discussions).
