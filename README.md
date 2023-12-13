# WhatsTheRating

## Overview

by Ansh Mujral and Keenan Serrao

Exploratory data analysis on this data can be found [here](https://keenans04.github.io/RecipesAndRatings/)!

## Framing the Problem

**-Prediction Problem** 

Using the Recipes and Ratings dataset, we will create a model that predicts the rating of a recipe. Using our exploratory data analysis from our previous report, we determined that the **minutes** that it takes to cook a recipe and the number of ingredients **(n_ingredients)** in a recipe contribute to the rating of a recipe. We will examine this correlation further and determine its effectiveness with the indicators below.

**-Classification or Regression?**

The model we are creating is a multiclass classification model because there are only 5 labels that can be classified. This is a classification model since we are trying to predict the correct label **(rating)** given the inputs of **minutes** and **n_ingredients**. Also, binary classification can not be applied in this scenario because we have more than 2 labels.

**-Response Variable**

Our response variable is **rating**. Rating is a quantitative and discrete variable with values of 1,2,3,4, and 5. Discrete quantitative variables are easier to classify as they can not have a decimal value, thus making it easier to predict a label for.

**-Metrics**

**Information Known**

## Baseline Model

**-Description** 

**-Features**

**-Performance**


## Final Model

**-Features Added** 

**-Description**

**-Modeling Algoroithm**

**-Hyperparameters**

**-Performance**

## Fairness Model

**Description**
Our group asks the question, “Does our final model perform better for recipes of low ratings (1,2) than it does for higher ratings (3,4,5)? To simulate this, we will be applying a permutation test where we shuffle both groups x and y. Our evaluation metric will be precision because we want to learn more about how our model identifies and classifies low and high-rated recipes, providing a nuanced understanding of its discriminatory performance across different rating categories.

**Null Hypothesis** Our model is fair. Its precision for low ratings and high ratings are roughly the same, and any differences are due to random chance.

**Alternate Hypothesis** Our model is unfair. Its precision for low ratings is lower than for high ratings

**Test Statistic** Absolute difference in precision

**Significance Level** 5% significance level

**P-value** 0.0

**Conclusion** 
Upon examining our distribution and the test statistic, we got a p-value of 0.0. Since the p-value of 0.0 is less than 0.05, our significance level, we **reject** the null hypothesis, and conclude that our model precision for low ratings is lower than it is for higher ratings. This permutation test turns out to be statistically significant **presumes that our model isn't fair** and **slightly biased** towards higher ratings. However, this is **not a 100% guarantee** that our model is biased and this test **does not mean an absolute conclusion**. 

Overall, this test does confirm our initial thoughts that the model could be biased, given the significant prevalence of outcomes favoring the higher ratings of 3,4, and 5, coupled with a lesser preference for the lower ratings of 1 and 2. Remarkably, the test statistic exhibited a pronounced skew towards the extreme, nearly overshadowing the overall distribution. This might be because there are not a lot of recipes with low ratings as food.com probably filters out lower rating recipes as it is not a good recipe to put on their site.

