# Winning solution to [*2016 Data Mining Cup*](http://www.data-mining-cup.de/en/review/goto/article/dmc-2016.html)
### Ranked [1/120](https://www.ucdavis.edu/news/uc-davis-statistics-students-win-international-data-competition) as Team Uni_UC_Davis_2


## Task description

The task of DMC 2016 is the prediction of return rates for a real anonymized fashion distributor, from October 2015 to December 2015, based on historical sales data and related return rates from January 2014 to September 2015. The training data consist of 2.33 million observations, and 14 predictors including 10 categorical variables and 4 numerical variables.

The dataset for DMC 2016 can be downloaded [here](http://www.data-mining-cup.de/en/review/goto/article/dmc-2016.html).

## Feature engineering

Feature engineering is always the most important and crutial part in data science competition. We approach the feature engineering problem from several different perspectives:

- Aggregation. We group data (e.g., price, quantity) by certain variables, such as orderID, customerID, articleID and orderDate. To each group of the data, we apply aggregate functions including min, max, mean, sum, number of elements, number of unique elements, etc. We then expand the summarized data by inserting them into each row. Here are some examples: total quantity per order, total number of orders per customer and mean recommended retail price per article.

- Decoding.
  - ColorCode is represented by four-digit numbers, where each digit has its own meaning, such as color, shade and pattern. Thus, it is more reasonable to use each digit of colorCode as features. 
  - SizeCode is represented in different units for different productGroup and thus be converted into a uniform unit.
  - Rrp (recommended retail price) reflects the price level of the items, such as cheap, regular and luxury. 
  - PaymentMethod can be manually classified into two groups: invoice and right-away.

- Customer behavior.
  - We propose some criteria to measure similarity between items in a single order. For example, items with the same articleID and colorCode, with the same productGroup and rrp, etc. For each item in a single order, we count the total quantity of items that are similar to it. Intuitively, the larger the value is, the higher probability of returning this item is.
  - We create features that reflect customer preferences since they tend to pick their preferred items among similar items. For example, the percentage of each possible color of a specific article bought by a customer. 
  - We extend the idea of similar items to the across-order cases. If a customer has bought similar items many times, the customer is very familiar with this type of items and has a higher probability of keeping it. We create relevant features based on these conjectures.

- Likelihood of returning. We use the out-of-sample returning rate to construct the likelihood features for each customer and each month. It turns out that the change of the prediction is significant when likelihood features are added, and this would greatly improve the performance of model stacking described later.

- New productGroup transformation. The new productGroups that appear in the test data only are manually imputed by matching them with the existing productGroups. We cannot validate this extrapolation using the available data. To be conservative, the final prediction is a weighted average of the predictions where the imputation is conducted in two different ways.

The script for feature engineering is *feature_data.R*. The package **data.table** is used to make the syntax concise and computation fast.

## Modeling strategy

- We reformulated the prediction task into a classification problem by expanding the data with respect to the predictor quantity, so that the new response on each line is either 0 or 1. The predicted probabilities are summed accordingly and rounded in the end to revert back to the original scale. Doing so reduces the multi-class classification problem to binary classification, where the computational cost of the latter is much lesser since there are fewer parameters to be estimated in the model. Although the evaluation criterion of the task is mean absolute error (MAE), we used logarithmic loss (log-loss) as the objective function to be optimized. Compared with MAE, which is non-smooth, log-loss is more stable and a better criterion to estimate class probabilities.

- With the derived features, we applied stacked generalization (Wolpert, 1992): a multi- layer modeling approach to combine the predictions of several base learners to improve the predictive power through leveraging the strength of each base model and to avoid overfitting. The outline (see Figure 4) is as follows:
1. Train base learners with the help of cross-validation, and predict for both training and test data. We used regularized logistic regression (glmnet) (Friedman et al., 2010), random forest (parRF) (Breiman, 2001), deep learning (h2o) (Arora et al., 2015) and gradient boosting (xgboost) (Chen and He, 2015) models, where the R packages we used are indicated in parenthesis.
2. Treat the predicted probabilities of return as new features. Combine them with the top 100 important features, and feed them into xgboost and deep learning to generate second layer predictions.
3. Bagging the second layer predictions to form the final prediction. In practice, we only used the boosting predictions due to their very high correlation with the deep learning predictions.


![alt text](https://github.com/jlyang1990/Data_Mining_Cup_2016/blob/master/images/flow_chart.pdf)

- It is very important to have a robust performance evaluation criterion for feature selection and model comparison. In time series predictor evaluation, a blocked form of cross-validation (Bergmeir and Ben ́ıtez, 2012) is more suitable than the traditional cross-validation since the former respects the temporal dependence. However, it suffers from the problem of predicting the past based on the future. Another common practice is to reserve a part from the end of each time series for testing, and to use the rest of the series for training. This strategy avoids predicting the past, but it does not make full use of the data, and thus is less stable than cross-validation and prone to be over-fitting to the test data. To validate the modeling strategy, we applied both methods. Specifically, we divided the training data into 7 cross- validation folds with about 3 months in each fold, and treated the last fold (called holdout set) as a pseudo test set.

## Acknowledgement

There are four other teammates, without whom the competition is impossible to be finished: Jilei Yang, Qi Gao, Nana Wang and Chunzhe Zhang. We would like to express our special thanks to Prof. Hao Chen, who provided us very insightful advice and practical guidance. We are also thankful for the support we get from the STA260 class and Prof. Wang, and also for Prudsys AG that held and sponsored this interesting competition.

Thanks Minjie Fan and Hao Ji for writing a detailed report to summarize our work in DMC 2016. Most of the descriptions in README are from this report. The report can be reached in [Minjie Fan's Github](https://github.com/minjay/DMC2016).

**At last, I would like to share one saying from a sucessful Kaggler with you: *''Features make difference and ensemble makes you win.''* This is exactly what we have learned from 2016 Data Mining Cup.**