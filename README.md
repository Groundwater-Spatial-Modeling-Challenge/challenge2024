# The Groundwater Spatial Modeling Challenge 2024

This challenge is strongly inspired by the last years **Groundwater Time Series Modeling Challenge** organized by R.A. Collenteur (Eawag) et al. and can be considered as an extension to the spatial domain.


> [!NOTE]
> **Update 2024/04/10: Data is released and the challenge has started !**


This repository contains all the information and materials for the Groundwater Spatial Modeling Challenge, [as 
announced at the 2024 EGU General Assembly](https://meetingorganizer.copernicus.org/EGU24/EGU24-10386.html).
We would like to invite every interested scientist or modeler to participate in this contest and submit the results of their best-performing models.
In this way, we not only want to learn from each other, but also bring shared experience and creativity to this still very small community.

**Organisers:**  E. Haaf (Chalmers), T. Liesch & M. Ohmer (KIT), M. Nölscher (BGR)


## Background & Objectives

Spatially continuous information (aka. maps) about groundwater related parameters is a crucial pre-condition for many tasks in water management and ecological questions.
However, in hydrogeology, we often lack such reliable datasets due to the fact that observed data comes almost always from observation wells which reflect only a very narrow area around it. Based on such observed data, spatially continuous datasets can be then generated using deterministic methods like Voronoi-polygons or Inverse-Distance-Weighting or geostatistical methods like the Krigging-family for inter- and extrapolating over space. In the recent two decades machine learning approaches have been increasingly studied and applied for this task. Compared to time series modeling of groundwater levels, there has not yet been a comparable breakthrough in mapping/regionalizing groundwater related parameters.
Whith this challenge, we want to 

- put more focus on this matter of research,
- grow the community,
- increase discussion,
- showcase the diversity of approaches and evaluate their capabilities and
- learn from each others experience and creativity.

In order to get a tiny bit closer to these goals, this challenge is about modeling nitrate concentrations (including the prediction intervals) in shallow aquifers in southwest Germany using a broad range of geophysical predictor variables aka. features. Any model is welcome. It is not restricted to machine learning models.

In the following, all necessary information on participating in the challenge is explained.


## Input and hydraulic head data

Five hydraulic head time series were selected for this challenge. The monitoring wells are located in sedimentary 
aquifers, but in different climatological and hydrogeological settings. Depending on the location. different input 
time series are available to model the heads. Please find all data and descriptions in the [data folder](https://github.com/gwmodeling/challenge/tree/main/data).

It is permitted to use any other publicly available data (e.g., soil maps) to construct the model. The use of other 
meteorological data that that provided is not permitted, to ensure that differences between the models are not the 
result of the meteorological input data. It is also not permitted to use the hydraulic heads as explanatory 
variables in the model.

## Modeling rules

- Anyone interested can participate as single person or as team
- Participants may use any type of model
- The groundwater time series themselves may not be used as model input
- The modeling workflow must be reproducible, preferably through the use of scripts, but otherwise described in 
  enough detail to reproduce the results. This requires freely available software, preferably open source
- Supplementary model data must be described in sufficient detail and submitted with model outputs.
- Submission of model results are done via a github pull request

## Model outputs and deliverables

The model is expected to compute: 

-	The prediction of the hydraulic head for the dates found in the submission files in the  'team_example' folder, 
    including the 95% prediction interval of the hydraulic head at a daily time interval over the entire 
     calibration and validation period (see data folders for specific periods for each location).

Forms that can be used to submit the results are provided in the [submissions folder](https://github.com/gwmodeling/challenge/tree/main/submissions). 
There you can also find more detailed on what to submit.

## Model evaluation

The models will be evaluated using several goodness-of-fit metrics and groundwater signatures, computed for both the 
calibration and the validation period. The data for the validation period is not make public yet and will be 
released after the challenge ended.

## Deadline

The deadline for the challenge is **31/12/2022. Late submission are allowed untill 5th of January 24:00 CET.** Please make sure to submit before this date. We plan to share the results of this challenge at the EGU General Assembly 2023.

## Participation & Submission
If you intend to participate, [please open a GitHub Issue for your team](https://github.com/gwmodeling/challenge/issues), such that we can track the participating teams.

Participant can submit their model results as a Pull Request to this Repository, adding a folder with their results 
in the 'submissions' folder. The model results must be submitted in a way that they are reproducible, either through 
the use of scripts (preferred) or detailed description of the modeling process. See the [submissions folder](https://github.com/gwmodeling/challenge/tree/main/submissions) for a more detailed description on how and what to submit.

After the challenge we intend to write an article to submit to a peer-reviewed journal with all the organisers and participants.

## Questions/ Comments ?

To make sure everyone has access to the same information we ask you to put any questions that are of general 
interest to all participants in the [GitHub Discussion forum](https://github.com/gwmodeling/challenge/discussions).



