---
layout: post
title: 'Machine Learning: Bias VS. Variance'
date: 'October 11th, 2018'
categories: datascience, machinelearning, bigdata
description: 'The difference between bias and variance in machine learning.'
tags: datascience, machinelearning, bigdata
permalink: mlbiasvsvariance.html
---

# Machine Learning: Bias VS. Variance

What is BIAS?

![](https://cdn-images-1.medium.com/max/9216/1*LN-6ANok1j6qCUo4sSyNtA.jpeg)

From EliteDataScience, bias is: “Bias occurs when an algorithm has limited flexibility to learn the true signal from the dataset.”

Wikipedia states, “… bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).”

**Bias is the accuracy of our predictions.**

* A high bias means the prediction will be inaccurate. Intuitively, bias can be thought as having a ‘bias’ towards people. If you are highly biased, you are more likely to make wrong assumptions about them. An oversimplified mindset creates an unjust dynamic: you label them accordingly to a ‘bias.’

Forman’s article summarized this:
> “Bias is the algorithm’s tendency to consistently learn the wrong thing by not taking into account all the information in the data (underfitting).”

## Trending AI Articles:
> # [1. Google will beat Apple at its own game with superior AI](https://becominghuman.ai/google-will-beat-apple-at-its-own-game-with-superior-ai-534ab3ada949)
> # [2. The AI Job Wars: Episode I](https://becominghuman.ai/the-ai-job-wars-episode-i-c18e932ff225)
> # [3. Introducing Open Mined: Decentralised AI](https://becominghuman.ai/introducing-open-mined-decentralised-ai-18017f634a3f)
> # [4. AI & NLP Workshop](https://becominghuman.ai/ai-nlp-workshop-7bc121986d61)

Thus, parametric algorithms are prone to high bias. A parametric algorithm is defined as, *“A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model. No matter how much data you throw at a parametric model, it won’t change its mind about how many parameters it needs.”*

A linear regression is an example of a parametric algorithm. These are easy to understand but not flexible to learn the underlying signal of the data. Thus, they are inaccurate for complex datasets.

Examples of high-bias algorithms include Linear Regression, Linear Discriminant Analysis, and Logistic Regression.

## What is VARIANCE?

![](https://cdn-images-1.medium.com/max/6488/1*PTaEY-0nSRVYrS4ZwjuPHQ.jpeg)

From EliteDataScience, the variance is: “Variance refers to an algorithm’s sensitivity to specific sets of the training set occurs when an algorithm has limited flexibility to learn the true signal from the dataset.”

Wikipedia states, “… variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).”

**Variance is the difference between many model’s predictions.**

* Unlike the analogy as before, we are implementing complicated models. Hence, any ‘noise’ in the dataset, might be captured by the model. A high variance tends to occur when we use complicated models that can overfit our training sets. For example, a variance can be thought as having different stereotypes based on different demographics.

For example, a complicated model might depict people’s name as a good predictor of our hypothesis. However, names are random and should not have any predictive power. In one dataset, people with the name ‘Alex’ can indicate they are likely to be criminals. However, in another dataset, people with the name ‘Alex’ can indicate they likely to be graduates. Hence, names should not be used as a predictive variable.

Forman’s described variance as:
> “Variance is the algorithm’s tendency to learn random things irrespective of the real signal by fitting highly flexible models that follow the error/noise in the data too closely (overfitting).”

## What is the TRADE-OFF?

![](https://cdn-images-1.medium.com/max/7872/1*6dPPV8kz8Lt3-TCPhexBXw.jpeg)

* If you have a simple model, you might conclude that every “Alex” are amazing people. This presents a High Bias and Low Variance problem. Your dataset is ‘biased’ towards people with the name Alex. Thus, most predictions will be similar, since you believe people with ‘Alex’ act a certain way.

* You attempt to fix the model. However, the model is too complicated. Your model has different results for different groups. Thus, Alex can be a wonderful person, a criminal, an athlete, and a scholar.

* You must find balance! The good thing, if you do Cross-Validation, you can train on many datasets and average their predictions.
> # *Unfortunately, you cannot minimize bias and variance.*

**Low Bias — High Variance:**
> A low bias and high variance problem is overfitting. Different data sets are depicting insights given their respective dataset. Hence, the models will predict differently. However, if average the results, we will have a pretty accurate prediction.

**High Bias — Low Variance:**
> The predictions will be similar to one another but on average, they are inaccurate.

![](https://cdn-images-1.medium.com/max/2000/1*p725FfM2K5q3HoDp16nRDA.png)

### Lessons From Andrew Ng’s Course:

If you have HIGH VARIANCE PROBLEM:

1. You can get more training examples because a larger the dataset is more probable to get a higher predictions.

1. Try smaller sets of features (because you are overfitting)

1. Try increasing lambda, so you can not overfit the training set as much. The higher the lambda, the more the regularization applies, for Linear Regression with regularization.

If you have HIGH BIAS PROBLEM:

1. Try getting additional features, you are generalizing the datasets.

1. Try adding polynomial features, make the model more complicated.

1. Try decreasing lambda, so you can try to fit the data better. The lower the lambda, the less the regularization applies, for Linear Regression with regularization.

### Reminders:

If a learning algorithm is suffering from high variance, getting more training data helps a lot. High variance and low bias means overfitting. This is caused by understanding the data to well. With more data, it will find the signal and not the noise.

## WANT MORE…

If so, I suggest following my Instagram page. I post summaries and thoughts on a book that I have and am currently reading.

## Instagram: [Booktheories](http://instagram.com/booktheories/), [Personal](https://www.instagram.com/alexxestevenn_/)

## Follow me on: [Twitter](https://twitter.com/alexguangaa), [GitHub](https://github.com/alexguanga), and [LinkedIn](https://www.linkedin.com/in/alexguanga)

*AND if you liked this article, I’ll appreciate it if you click on the like button below. THANKS!*
