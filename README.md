# BigData_Project1
_Shien-Ming Wu School of Intelligent Engineering_
_South China University of Technology, Guangzhou, China_


This is the project 1 of the lesson Big Data and its Application.

Attention: All the code is written in _Python3_. If you want to run the code, make sure that you have set the right path in the _Import Data_ part of each code.

## Brief Introduction
LoL is one of the most popular online game currently around. It is a 5 vs. 5 competitive game. The target of the game is to destroy the base of the enemy. The team which grabs more resources, such as Dragon and Baron. 

Predicting its result from the game status is the target of this project. Now a dataset containing approximately 5 million game records is given. About 3 million is used for training and 2 million for testing. The record of a specific game includes the creation time of the game, the game duration, the ID of the season, the winner team, the team which gets first blood, the first tower, the first inhibitor and so on.

Some classifiers, which use information other than the winner to predict the winner, are expected to obtain. The algorithms used in this project include Decision Tree (DT), Support Vector Machine (SVM), Multi-Layer Perceptron (MLP), Random Forest (RF) and Voting Ensemble. In addition, Grid Search algorithm is used to get the optimized parameters for some classifiers.
