---
layout: post
title: 'Bias VS. Variance'
date: '2017-12-01 0:00:00 -0400'
categories: learning
description: 'Understanding bias and variance tradeoff. This is important to understand with figuring out the models to apply.'
tags: learning, statistics
---


# Bias VS. Trade-off

The Bias VS. Variance tradeoff is a simple yet weird concept to wrap your head around. This article will be used as a reference for myself. If you would like a more detailed tutorial, I recommend the following:

[Andrew Ng's Tutorial]('https://www.youtube.com/watch?v=e3edL-_fUTo')

[Scott Forman's Article]('http://scott.fortmann-roe.com/docs/BiasVariance.html')


#### What does BIAS mean?
Essentially bias is defined by the accuracy of the predictions. A higher bias means the prediction is further from the reality. Hence, a lower bias means you are the prediction is closer to reality. An analogy is the following: **If you bias towards a certain group of people, you are more likely to assume wrong things about them. This might be because of misleading information or internal biases.** Whatever the case is, a higher bias means more inaccurate information.


When we have a large bias, we are generalizing our model to fit a dataset. Like the example mentioned, if you generalized a group of people, you are more likely to make a false assumption about them. Also, you are less likely to pick any information on specific people if you are generalizing an entire group.


Forman's article summarized this, *"Bias is the algorithm's tendency to consistently learn the wrong thing by not taking into account all the information in the data (underfitting)."*


A linear regression is an example of a parametric algorithm. These are easier to understand but are not flexible to depict the signal. In turn, they have lower predictive performance on complex problems that fail to meet the simplifying assumptions of the algorithms bias.


Examples of high-bias machine learning algorithms include Linear Regression, Linear Discriminant Analysis, and Logistic Regression.


#### What does VARIANCE mean?
Variance is the difference between the application of your model on different datasets. Following the same analogy as before, imagine if you're using a simple model (linear regression) to understand a certain group of people. For the most part, the results will be the same. There is not much going on with an algorithm for each to change among subgroups.


However, if the model is more complicated, the results will differ among different groups. It some cases, it might capture the noise instead of the signal. Thus, this means there's a high variance. A high variance tends to occur when we have complicated models that overfit its specific its training sets and performs badly for the testing sets.


*"Variance is the algorithm's tendency to learn random things irrespective of the real signal by fitting highly flexible models that follow the error/noise in the data too closely (overfitting)."*


Meaning, if you trained your dataset and found that people with the name "Alex" are incredible people because that was your sample size, then when you used that algorithm to train the test set, you will get the wrong prediction. Hence, your algorithm picked up too much of the noise, it is overfitting for that specific data.


#### Where is does TRADE-OFF?
Think about it.
1. If you have a simple and general model, you might conclude that all people with the name Alex are amazing. This will present a High Bias problem.
2. As you want to fix your model, you have eventually complicated it a lot. Now, your model gives different results among different groups. Thus, Alex can be a wonderful person, a criminal, an athlete, and a scholar.
3. Thus, you must find a balance! The good thing, if you do Cross-Validation, you can train your data a lot of times and get the average among all the tries.


You cannot minimize both at the same time. These are the choices.
1. Low Bias - High Variance: For this case, on average, be close to the prediction but with each individual case different from one another. Meaning, I might overfit the dataset a bit but on average, I will get a close prediction of the actual.
2. High Bias - Low Variance. For this case, you can have your data far from the prediction but each case close to one another.

![alt text](https://www.dropbox.com/s/s71ggnyqj7fhhhd/Bias-Variance-2.png?raw=1 "Logo Title Text 1")

![alt text](https://www.dropbox.com/s/so8uf5ucrhmmlbx/Image8.png?raw=1 "Logo Title Text 1")

#### Lessons From Andrew Ng's Course:

**If you have HIGH VARIANCE PROBLEM:**

1. You can get more training examples because of the larger the dataset, the greater the possibility to find the actual signal.
2. Try smaller sets of features (because you are overfitting)
3. Try increasing lambda, so you can not overfit the training set as much. The higher the lambda, the more the regularization applies, for Linear Regression with regularization.

**If you have HIGH BIAS PROBLEM:**

1. Try getting additional features, you are generalizing the datasets.
2. Try adding polynomial features, make the model more complicated.
3. Try decreasing lambda, so you can try to fit the data better. The lower the lambda, the less the regularization applies, for Linear Regression with regularization.


#### Reminders:
If a learning algorithm is suffering from high variance, getting more training data WILL helps a lot. Variance means overfitting. This is caused by understanding the given data to well. With more data, it will find the signal and not the noise. Likewise, more data does not help a High Bias situation.
