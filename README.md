This is a personal project and my first attempt at building my first machine learning classification model.

The goal of this project was to give myself real life exposure using a real dataset to clean and perform analysis, leading into building a prediction model.

comp_data = Processed data that contains both features and targets. This dataset will be used to manipulate and train models

ml_data = Manipulated comp_data that maps medication to numeric values, drops race, and any ID

race_data = ml_data but contains Race for appropriate further breakdown into subsets for individual races

Model comparison using AUROC, AUPRC, and matrices to visualize true positive/false positives