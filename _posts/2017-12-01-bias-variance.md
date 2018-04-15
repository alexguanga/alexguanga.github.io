---
layout: post
title: 'Bias VS. Variance'
date: '2017-12-01 0:00:00 -0400'
categories: learning
description: 'Understanding bias and variance tradeoff. This is important to understand with figuring out the models to apply.'
tags: learning, statistics
permalink: biasvariance.html
---


# Bias VS. Trade-off

The Bias VS. Variance tradeoff is a simple yet weird concept to wrap your head around. This article will be used as a reference for myself. If you would like a more detailed tutorial, I recommend the following:

[Andrew Ng's Tutorial]('https://www.youtube.com/watch?v=e3edL-_fUTo')

[Scott Forman's Article]('http://scott.fortmann-roe.com/docs/BiasVariance.html')


#### What is BIAS?

From EliteDataScience, bias is: *"Bias occurs when an algorithm has limited flexibility to learn the true signal from the dataset.”*

Wikipedia states, "... bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).”

Bias is the accuracy of our predictions. A high bias means the prediction is inaccurate and vice versa. More intuitively, bias can be thought a 'bias' towards a group of people. If you are highly biased, you are more likely to make wrong assumptions about them. An oversimplified mindset creates an unjust dynamic: you think of them accordingly to your 'bias.’

Forman’s article summarized this, “Bias is the algorithm’s tendency to consistently learn the wrong thing by not taking into account all the information in the data (underfitting).”

Thus, parametric algorithms are prone to high bias. A parametric algorithm is defined as, "A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model. No matter how much data you throw at a parametric model, it won’t change its mind about how many parameters it needs."

A linear regression is an example of a parametric algorithm. These are easy to understand but not flexible to learn the underlying signal of the data. Thus, they are inaccurate for complex datasets.

Examples of high-bias machine learning algorithms include Linear Regression, Linear Discriminant Analysis, and Logistic Regression.


#### What is VARIANCE?

From EliteDataScience, variance is: *"Variance refers to an algorithm's sensitivity to specific sets of the training set occurs when an algorithm has limited flexibility to learn the true signal from the dataset.”*

Wikipedia states, "... variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting)."

Variance measures our model's prediction on different models. Unlike the analogy as before, we are implementing complicated models. Hence, any 'noise' in the dataset, might be captured by the model. A high variance tends to occur when we use complicated models that can overfit our training sets. Using stereotypes, the variance can be thought as having a model for different groups of people.

For example, a complicated model might depict people's name as a good predictor of our hypothesis. However, names are random and should not have any predictive power. In one dataset, people with the name 'Alex' can indicate they are likely to be criminals. However, in another dataset, people with the name 'Alex' can indicate they likely to be graduates. Hence, names are not a great variable to be used but give our complicated models, it believes that a name can be a predictor.

Forman’s described variance as, “Variance is the algorithm’s tendency to learn random things irrespective of the real signal by fitting highly flexible models that follow the error/noise in the data too closely (overfitting).”

#### What is the TRADE-OFF?

- If you have a simple model, you might conclude that every "Alex" are amazing people. This presents a High Bias and Low Variance problem. Your dataset is 'biased' towards people with the name Alex. Thus, most predictions will be similar, since you believe people with 'Alex' act a certain way.

- You attempt to fix the model. However, the model is too complicated. Your model has different results for different groups. Thus, Alex can be a wonderful person, a criminal, an athlete, and a scholar.

- You must find balance! The good thing, if you do Cross-Validation, you can train on many datasets and average their predictions.

Unfortunately, you cannot minimize bias and variance.

1. Low Bias - High Variance: A low bias and high variance problem is overfitting. Different data sets are depicting insights given their respective dataset. Hence, the models will predict differently. However, if average the results, we will have a pretty accurate prediction.


2. High Bias - Low Variance: The predictions will be similar to one another but on average, they are inaccurate.

************************************************************************************

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
