# WhatsTheRating

## Overview

by Ansh Mujral and Keenan Serrao

Exploratory data analysis on this data can be found [here](https://keenans04.github.io/RecipesAndRatings/)!

## Framing the Problem

**-Prediction Problem** 

Using the Recipes and Ratings dataset, we will create a model that predicts the rating of a recipe. Using our exploratory data analysis from our previous report, we determined that the `minutes` that it takes to cook a recipe and the `calories` of a recipe contribute to the `rating` of a recipe. We will examine this correlation further and determine its effectiveness with the indicators below.

**-Classification or Regression?**

The model we are creating is a multiclass classification model because there are only 5 labels that can be classified. This is a classification model since we are trying to predict the correct label `(rating)` given the inputs of `minutes` and `calories`. Also, binary classification can not be applied in this scenario because we have more than 2 labels.

**-Response Variable**

Our response variable is `rating`. Rating is a quantitative discrete and a categorical ordinal variable with values of 1,2,3,4, and 5. Discrete quantitative variables are easier to classify as they can not have a decimal value, thus making it easier to predict a label. Regression can not be used on categorical data.

**-Metrics**

The most suitable metric for predicting ratings would be F-1 Score. This is because our dataset is inbalanced as there are a lot more higher ratings than lower ratings. Additionally, we want to limit false positives and negatives. Recipes should have equal value ratings as low-rating recipes may not be cooked and overly high-rated recipes may get overly critiqued. We are prioritizing avoiding overrating (false positives) and underrating (false negatives) recipes, especially since user satisfaction is our primary concern. This means both high recall and high precision is necessary and the F-1 score is a great estimator of both.

**Information Known**

At the time of prediction, `minutes` (quantitative continuous), `number of ingredients` (quantitative discrete), `nutritional values` (quantitative continuous), and `number of steps` (quantitative discrete) are the information we know. Each recipe would have needed to be cooked and tasted before someone could submit a rating. This means that the ingredients used, the nutritional values of those ingredients used, the cooking time, and the number of steps it took to prepare the food would all be known at the time of prediction. 

## Baseline Model

**-Description** 

Our baseline model currently uses 2 feature predictors, `minutes` (quantitative continuous) and `calories` (quantitative continuous). We will use these two features to predict `ratings` (categorical discrete) in our Decision Tree Classification model.  

**-Features**

Our baseline model currently uses 2 feature predictors, `minutes` and `calories`. Both of these features are quantitative continuous and are not categorical, so there was no need to apply one hot encoding, ordinal encoding, or any other type of feature engineering to the model. By maintaining the raw, unaltered values of calories and minutes, we intentionally preserved the simplicity of our model. This deliberate choice allowed us to gain valuable insights into the classifier's performance in predicting ratings solely based on these fundamental elements, providing a clear and unadulterated perspective on the interplay between cooking time, nutritional content, and recipe ratings.

**-Performance**

We **do not** believe our current model is good because it always predicts a 5-star rating for each recipe. This is because there is a substantially larger amount of 5-star recipes than every other recipe. The F-1 Score calculated for the test data of this model is 0.699, which is very bad. This is probably due to the huge imbalance in 1-star, 2-star, 3-star, 4-star, and 5-star reviews. Below we have the confusion matrix showing the number of actual positive and predicted positive values.


## Final Model

**-Features Added** 

**-Description**

**-Modeling Algoroithm**

After trying multiple classifiers such as KNNclassifier, RandomForestClassifier, and Naive Bayes, we chose to use a Decision Classifier. We chose to use a Decision Classifier due to its interpretability, robustness against outliers and irrelevant values (which this dataset has, shown in the EDA), and its quickness in fitting and predicting a model. 

Although Decision Classifiers are known to overfit data, we have used GridSearchCV to find the best possible hyperparameters to maximize our test set. We also made sure to stop splitting after a certain depth (mentioned more in the hyperparameters) to avoid overfitting and to limit the high variance that Decision Classifiers can be known to achieve.

**-Hyperparameters**

Through our implementation of GridSearchCV, we meticulously optimized the hyperparameters, specifically max_depth and criterion, in our Decision Tree Classifier. This strategic fine-tuning aimed to discover the most effective parameter combination that enhances the classifier's ability to generalize well to previously unseen data (test data). The best hyperparameters we found were **max_depth** = 3, and **criterion** = 'gini'.

**-Performance**

## Fairness Model

**Description**
Our group asks the question, “Does our final model perform better for recipes of low ratings (1,2) than it does for higher ratings (3,4,5)? To simulate this, we will be applying a permutation test where we shuffle both groups x and y. Our evaluation metric will be an F-1 Score because we want to learn more about how our model identifies, classifies, overrates, and underrates low and high-rated recipes, providing a nuanced understanding of its performance across different rating categories.

**Null Hypothesis** Our model is fair. Its precision for low ratings and high ratings are roughly the same, and any differences are due to random chance.

**Alternate Hypothesis** Our model is unfair. Its precision for low ratings is lower than for high ratings

**Test Statistic** Absolute difference in precision

**Significance Level** 5% significance level

**P-value** 0.0

**Conclusion** 
Upon examining our distribution and the test statistic, we got a p-value of 0.0. Since the p-value of 0.0 is less than 0.05, our significance level, we **reject** the null hypothesis, and conclude that our model F-1 score for low ratings is lower than it is for higher ratings. This permutation test turns out to be statistically significant **presumes that our model isn't fair** and **slightly biased** towards higher ratings. However, this is **not a 100% guarantee** that our model is biased, and this test **does not mean an absolute conclusion**. 

Overall, this test does confirm our initial thoughts that the model could be biased, given the significant prevalence of outcomes favoring the higher ratings of 3,4, and 5, coupled with a lesser preference for the lower ratings of 1 and 2. Remarkably, the test statistic exhibited a pronounced skew towards the extreme, nearly overshadowing the overall distribution. This might be because there are not a lot of recipes with low ratings as food.com probably filters out lower rating recipes as it is not a good recipe to put on their site.

