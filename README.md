# [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

[github pages](https://ptiwaree.github.io/Kaggle-House-Prices/)

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

The potential for creative feature engineering provides a rich opportunity for fun and learning. This dataset lends itself to advanced regression techniques like random forests and gradient boosting with the popular XGBoost library. We encourage Kagglers to create benchmark code and tutorials on Kernels for community learning. Top kernels will be awarded swag prizes at the competition close. 


## Introduction

In this Kaggle challenge, I use different Regression and other Machine Learning techniques to predict the final price for each home using the features provided in the dataset. Before running ML algorithms, I preprocessed the data to remove outliers, to deal with missing values, and to encode categorical variables. 

## Data Preprocessing and Exploration

[Data Preprocessing Notebook](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Modeling/House%20Prices%20-%20Feature%20Engineering-2.ipynb)

In the above notebook, I do the following:

  1. Load the csv files
  2. Look into null values and fix them if they needed fixing. Some nulls were legitimate. For example, if the house has no fireplace (Fireplaces=0), then Fireplace Quality (FireplaceQu) equal to null is fine. In other cases, like LotFrontage, I use the median LotFrontage for the neighborhood to replace the nulls.
  3. Address skewness of features and output by log transforming them.
  4. Do one-hot encoding for categorical variables. This increased the number of features but not by much.
  
Some of the features are shown below in this plot of features vs output price

![Features](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Results/Features.png)
  
## Model Selection

I decided to use Ridge and Lasso Regression, Gradient Boosting Machine (GradientBoostingRegressor), and ExtraTreesRegressor which are all part of sklearn package to predict the output. Ridge, Lasso and GBM had similar Cross Validation Score (using 5-fold) while ExtraTreesRegressor did worse so I stacked the first 3 algorithms using xgboost. 

## Model Training

We can improve a model’s performance by tuning its parameters. So I used grid search method to tune hyperparameters for xgboost regressor, GradientBoostingRegressor, and ExtraTreesRegressor. Following notebooks contain Model Training.

[Boosting](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Modeling/House%20Prices%20-%20Boosting-3.ipynb)
[ExtraTreesRegression](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Modeling/House%20Prices%20-%20ExtraTreesRegressor.ipynb)

## Ensemble Generation

Ensemble Learning refers to the technique of combining different models. It reduces both bias and variance of the final model, thus increasing the score and reducing the risk of overfitting. Techniques like boosted trees and ExtraTreesRegressor are already using Ensemble method and I also used stacking method to combine the output from the 3 techniques (Ridge, Lasso and GBM) and used them as meta features to xgboost algorithm to predict final output. Following notebook has the details:

[Stacking Notebook](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Modeling/House%20Prices%20-%20Stacking.ipynb)

## Results

My submission is currently ranked in the Top 12% on Kaggle and this can certainly be improved. One thing I notice is that at higher house prices errors increase and I seem to be underpredicting the price. Following charts show this (it is predicted on 20% of the validation data set aside from the Training data):

![Predictions1](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Results/predictionchart1.png)

![Predictions2](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Results/predictionchart2.png)

![Training Predictions](https://github.com/ptiwaree/Kaggle-House-Prices/blob/master/Results/predictiontrain2.png)

## Tools Utilized

####Stack:

* python
* git

####Modeling:

* numpy
* scipy
* pandas
* scikit-learn
* xgboost

####Plotting:

* matplotlib
* seaborn
