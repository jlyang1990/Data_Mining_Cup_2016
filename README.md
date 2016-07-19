# Winning Solution to [*2016 Data Mining Cup*](http://www.data-mining-cup.de/en/review/goto/article/dmc-2016.html)
### Ranked [1/120](https://www.ucdavis.edu/news/uc-davis-statistics-students-win-international-data-competition) as Team Uni_UC_Davis_2


## Task description

The task of DMC 2016 is the prediction of return rates for a real anonymized fashion distributor, from October 2015 to December 2015, based on historical sales data and related return rates from January 2014 to September 2015. The training data consist of 2.33 million observations, and 14 predictors including 10 categorical variables and 4 numerical variables.

The dataset for DMC 2016 can be downloaded [here](http://www.data-mining-cup.de/en/review/goto/article/dmc-2016.html).

## Feature engineering

Feature engineering is always the most important and crutial part in data science competition. We approached the feature engineering problem from several different perspectives:

- Aggregation. We grouped data (e.g., price, quantity) by certain variables, such as `orderID`, `customerID`, `articleID` and `orderDate`. To each group of the data, we applied aggregate functions including mean, sum, number of elements, number of unique elements, etc. We then expanded the summarized data by inserting them into each row. Here are some examples: total quantity per order, total number of orders per customer and mean recommended retail price per article.

- Decoding.
  - `ColorCode` is represented by four-digit numbers, where each digit has its own meaning, such as color, shade and pattern. Thus, it is more reasonable to use each digit of colorCode as features. 
  - `SizeCode` is represented in different units for different product groups and thus is converted into a uniform unit.
  - `rrp` (recommended retail price) reflects the price level of the items, such as cheap, regular and luxury. 
  - `PaymentMethod` can be manually classified into two groups: invoice and right-away.

- Customer behavior.
  - We proposed some criteria to measure similarity between items in a single order. For example, items with the same `articleID` and `colorCode`, with the same `productGroup` and `rrp`, etc. For each item in a single order, we counted the total quantity of items that are similar to it. Intuitively, the larger the value is, the higher probability of returning this item is.
  - We created features that reflect customer preferences since they tend to pick their preferred items among similar items. For example, the percentage of each possible color of a specific article bought by a customer. 
  - We extended the idea of similar items to the across-order cases. If a customer has bought similar items many times, the customer is very familiar with this type of items and has a higher probability of keeping it.

- Likelihood of returning. We used the out-of-sample returning rate to construct the likelihood features for each customer and each month. It turns out that the change of the prediction is significant when likelihood features are added, and this would greatly improve the performance of model stacking described later.

- New product group transformation. The new product groups that appear in the test data only are manually imputed by matching them with the existing product groups. To be conservative, the final prediction is a weighted average of the predictions where the imputation is conducted in two different ways.

The script for feature engineering is [*feature_data.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/feature_data.R). The package **data.table** is used to make the syntax concise and computation fast.

## Modeling strategy

- We reformulated the prediction task into a binary classification problem by expanding the data with respect to the predictor quantity, so that the new response on each line is either 0 or 1. The computational cost is much less compared with the multi-class classification problem.

- We applied stacked generalization: a multi-layer modeling approach to combine the predictions of several base learners to improve the predictive power through leveraging the strength of each base model and to avoid overfitting. The outline is as follows:
  1. Train base learners with the help of cross-validation, and predict for both training and test data. We used regularized logistic regression (glmnet), random forest (parRF), deep learning (h2o.deeplearning) and gradient boosting (xgboost) models, where the R packages we used are indicated in parenthesis.
  2. Treat the predicted probabilities of return as new features. Combine them with the top 100 important features, and feed them into xgboost to generate second layer predictions.
  3. Bagging the second layer predictions to form the final prediction.

- In time series predictor evaluation, a blocked form of cross-validation is more suitable than the traditional one since the former respects the temporal dependence. However, it suffers from the problem of predicting the past based on the future. Another common practice is to reserve a part from the end of time series for testing, and to use the rest for training. This strategy avoids predicting the past, but it does not make full use of the data. To validate the modeling strategy, we combined these two methods. Specifically, we divided the training data into 7 cross-validation folds with 3 months in each fold, and treated the last fold (called holdout set) as a pseudo test set. For the final prediction, we trained the final model on all these 7 cross-validation folds and predicted on the test set.

The scripts for base learner training (first layer prediction) are [*model_xgb.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/model_xgb.R), [*model_dl.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/model_dl.R), [*model_glm.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/model_glm.R) and [*model_rf.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/model_rf.R).

The scripts for model stacking (second layer prediction) are [*stacking_data.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/stacking_data.R) and [*stacking_xgb.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/stacking_xgb.R).

The script for final prediction generation (third layer prediction) is [*final_result.R*](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/final_result.R).

## Acknowledgement

There are five other teammates, without whom the competition is impossible to be finished: Minjie Fan, Hao Ji, Qi Gao, Nana Wang and Chunzhe Zhang.

We would like to express our special thanks to Prof. Hao Chen, who provided us with very insightful advice and practical guidance. We are also thankful for the support we get from the Department of Statistics at UC Davis, and also for Prudsys AG that held and sponsored this interesting competition.

I would personally thank Minjie Fan and Hao Ji for writing a detailed report to summarize our work in DMC 2016. Most of the descriptions in README are from this report. The report can be reached in [Minjie Fan's Github](https://github.com/minjay/DMC2016).

**At last, I would like to share one saying from a sucessful Kaggler with you: *''Features make difference and ensemble makes you win.''* This is exactly what we have learned from 2016 Data Mining Cup.**