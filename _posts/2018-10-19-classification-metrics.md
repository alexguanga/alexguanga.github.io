---
layout: post
title: 'Understand Classification Performance Metrics'
date: 'October 19th, 2018'
categories: datascience, machinelearning, bigdata
description: 'Better understand how to access your classification models.'
tags: datascience, machinelearning, bigdata
permalink: classificationmetrics.html
---

# Understand Classification Performance Metrics

You don’t always want to be “accurate”…

![](https://cdn-images-1.medium.com/max/4586/1*EbpKczfURCvAF6uUp_Hhew.jpeg)

You have been working on a data science project. You have cleaned the data. You visualized the data. You structured the data. You understand your responsibility.

You know that your prediction can either be “yes” or “no.” Thus, you have two possible outcomes. You are dealing with a classification problem.

You have some inputs (variables). Your task is to create a model that could predict yes or no based on the data. You have tried a few different models. You might have tried various iterations of Logistic Regression, Support Vector Machines, and Random Forest Classification.

## Trending AI Articles:
> # [1. Logging in Tensorboard with PyTorch (or any other library)](https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e)
> # [2. A Short Machine Learning Explanation](https://becominghuman.ai/a-short-machine-learning-explanation-in-terms-of-linear-algebra-probability-and-calculus-f7660aa4b06c)
> # [3. Natural vs Artificial Neural Networks](https://becominghuman.ai/natural-vs-artificial-neural-networks-9f3be2d45fdb)
> # [4. AI & NLP Workshop](https://becominghuman.ai/ai-nlp-workshop-7bc121986d61)

A critical concept before explaining classification metrics is how the process works. You have three datasets. They are a training set, validation set, and testing set.

1. **Training set**: You build your model using the data from the training set. You, models, learn from the inputs in this dataset.

1. **Validation set**: There are several ways you can obtain your validation set. A simple method is removing a portion of the data and making this your validation set. Thus, when you train the model, the model has not learned from the validation set. The validation set indicates how well your training set has performed. You can also tweak hyperparameters with the validations set since you know the correct answers.

1. **Testing set**: The testing set is typically nowhere near the data scientists. Testing our model during this steps indicates how well our model performs.

I created a random dataframe in Jupyter Notebook. We are in the validation set. Our client will provide us with a new dataset (testing set). Our objective is to have the best performance.
> # What do you mean by performance?

I’ll demonstrate how we classification models measure performances. There are nuances when deciding on how to measure performance. But first, let’s create a dataframe of 0’s and 1’s.

We also need to know these definitions:

1. True Positives: The total number of accurate predictions that were “positive.” In our example, this is the total number of correctly predicting an email as spam.

1. False Positives: The total number of inaccurate predictions that were “positive.” In our example, this is the total number of incorrectly predicting an email as spam.

1. True Negative: The total number of accurate predictions that were “negative.” In our example, this is the total number of correctly predicting an email as non-spam.

1. False Negative: The total number of inaccurate predictions that were “negative.” In our example, this is the total number of incorrectly predicting an email as non-spam.

To get an excellent visual on these metrics, we will use a Confusion Matrix. In Python, the sklearn library makes creating a Confusion Matrix very easy.

![](https://cdn-images-1.medium.com/max/2000/1*rpvtED63QpDxPaLotrzQlA.png)

If we refer back to the definitions:

1. True Positives: 238

1. False Positives: 19

1. True Negatives: 219

1. False Negatives: 24
> # “This is great Alex, but have not yet discussed performance!”

Correct. But we now have the information to proceed. I’ll explain Accuracy, Precision, Recall (sensitivity), F1-Score, Specificity, Log-Loss, and the ROC/AUC Curve.

![](https://cdn-images-1.medium.com/max/8580/1*M43BfNUcYwowp-MTQ_si5A.jpeg)

## **Accuracy**

Formula:

* # of correct/# of predictions

* (TP + TN)/(TP + FP + TN + FN)

Accuracy seems like it could be the best method. In our example, our accuracy would be 91.4%. A very high score. So why not use it?

My thoughts:

* Accuracy only works when both possible outcomes (email being spam or not) is equal. For example, if we have a dataset where 5% of their emails are spam, then we could follow a less sophisticated model and have better accuracy score. We could predict every email as non-spam and achieve a 95% accuracy score. The imbalance dataset makes accuracy, not a reliable performance metric to use. The paradox explained is refer as “[Accuracy Paradox](https://towardsdatascience.com/accuracy-paradox-897a69e2dd9b),”

## **Precision/Recall**

These two performance metrics are often use in conjunction.

**Precision**

Formula:

* TP / (TP + FP)

With precision, we are evaluating our data by its performance of ‘positive’ predictions. Our precision for the email example is 92.61%! Remember that ‘positive’ in our example is predicting an email is spam.

Think of precision of how precision positive predictions were.

**Recall (also called sensitivity)**

Formula:

* TP / (TP + FN)

With recall, we are evaluating our data by its performance of the ground truths for positive outcomes. Meaning, we are judging how well predicted positive when the result was *Positive*. Our recall for the email example is 90.83%!

My thoughts:

* You cannot have the best of both worlds. In our email example, we had high precision and recalled scores. However, there’s a trade-off. Think about it. The denominator of precision is TP + FP. It does not consider FN. Thus, if we only made one positive prediction, and were correct, then our precision score would be 1. Even though, we might have missed a lot of actual positive observations.

* On the other hand, Recall’s denominator is TP + FN. It does not look at all the confident predictions I made. We can predict that all emails are spam. Thus, we will have a Recall score of 1 since will are anticipating every example to be positive and our FN will be 0 (we are not prediction non-spam).

Moreover, it depends on the situation you are that dictates whether you should use precision or recall. In some scenarios, you are better off with high recall.

* If you label every email to be spam, you will have a recall of 1. Why? Because you are predicting an email is spam. There’s no way you can use False Negatives since you are not predicting that an email would be non-spam. However, you will also be incorrectly predicting spam emails as non-spam.

* If you predict that one email is spam, and you get that right, then you’re precision would be 1. Why? Because there are not false positives. However, you will be missing a lot of predictions.

There’s also a less-common performance metric: **Specificity**.

Specificity is the opposite of Sensitivity or Recall. Hence, the formula is TN/(TN+FP).

## F1-Score
> [“F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution.”](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/) — Renuka Joshi

Formula:

* (2 * (Precision * Recall))/(Precision + Recall)

The F1-Score is the weighted average ([harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)) of precision and recall. Our F1-Score would be 91.71%.

## **ROC curve**/AUC Score
> [An **ROC curve** (**receiver operating characteristic curve**) is a graph showing the performance of a classification model at all classification thresholds.](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
> [**AUC** stands for “Area under the ROC Curve.” That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

The ROC Curve/AUC Score is most useful when we are evaluating a model to itself. Let me explain. Our ground truths and predictions are 1’s and 0’s. However, our forecast is never a 1 or a 0. Instead, we predict a probability and then assessed it whether or not it’s 1 or a 0. Typically, the classification threshold is .5 (in the middle). But we could have better-performing classification threshold.

The metrics our ROC Curve looks at is TPR (True Positive Rate) and FPR (False Positive Rate).

* True Positive Rate: TP / (TP + FN)

* False Positive Rate: FP / (FP + TN)

![Source: [https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)](https://cdn-images-1.medium.com/max/2000/1*4OqtyBCCcD_89od_8fNkCg.png)*Source: [https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)*

Our goal with the ROC is for our blue line in the graph below to be as close as possible to upper-right portion of the figure. On the other hand, the closer it is to the linear line, the worst our model performance. The red dot represents the trade-off between TPR and FPR.

* If you are dealing with cancer patients, it’s more important to have a high TPR than an FPR. You want to detect as many people as possible with cancer accurately. It does not matter you have a high False Positive Rate.

* If you need to keep the False Positive Rate low, say you do not want to be biased against certain people, you must also accept a lower True Positive Rate.

![](https://cdn-images-1.medium.com/max/2000/1*6YpIZ5SF_Z6YOM3eeW8ZHg.png)

The AUC is the Area-Under-the-Curve. Hence, it’ll be a number between 0 and 1. We would like it for to be as close as 1.

I know this example was not useful in comparing models, but it’s important to know!

## Log-Loss
> [**Log Loss** takes into account the uncertainty of your prediction based on how much it varies from the actual label. This gives us a more nuanced view into the performance of our model.](http://wiki.fast.ai/index.php/Log_Loss)

Formula:

* *-(ylog(p) + (1-y)log(1-p))*

![](https://cdn-images-1.medium.com/max/2232/1*EYQJvq4ganClRSn3MmD9NA.png)

With the log-loss, we are using the probability of our prediction.

Let’s look at the graph above. Notice how it states when “true label = 1.” Well, that indicates the ground truths. Also, the slope of the figure is significant. When the predicted probability is closer to 0, our log loss grows exponentially. While at the other end, the log loss is close to 0.

The intuition also applies when the “true label = 0” except the graph is a horizontal reflection where we see the exponential growth at 1 and not 0.

Now, it will like to evaluate the performance across every classification label (Multi-classes) and every example, the formula is the following:

![**Multi-class logarithmic loss**](https://cdn-images-1.medium.com/max/2000/1*PhBwmikmI6FufXHbbIwiuA.png)***Multi-class logarithmic loss***

**Conclusion**:

I hope you gained an intuitive understanding of these various performance metrics. Just work on data science project in Kaggle to gain practical understanding!

## WANT MORE…

If so, I suggest following my Instagram page. I post summaries and thoughts on a book that I have and am currently reading.

## Instagram: [Booktheories](http://instagram.com/booktheories/), [Personal](https://www.instagram.com/alexxestevenn_/)

## Follow me on: [Twitter](https://twitter.com/alexguangaa), [GitHub](https://github.com/alexguanga), and [LinkedIn](https://www.linkedin.com/in/alexguanga)
