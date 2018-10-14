---
layout: post
title: 'Specificity & Sensitivity'
date: 'December 4th, 2017'
categories: learning
description: 'Information on specificity & sensitivity (true positive rate and true negative rate).'
tags: learning, articles, datascience
permalink: Specificity&Sensitivity.html
---


# Specificity VS. Sensitivity


|        n=200  | Prediction: True | Prediction: False|
| -------------       |:-------------:| :-----:|
| **Actual: True**  | 50      | 10 |
| **Actual: False** | 5      |   100 |

+ **True Positives (TP):** 50
+ **True Negatives (TN):** 100
+ **False Positives (FP):** 5
+ **False Negatives (FN):** 10


### Sensitivity Test
+ Formula: True positive rate (or sensitivity): TPR=TP/(TP+FN)
+ 50/(50+10) = 83.33%


The sensitivity of a test (also called True Positive Rate) is the proportion of people with a disease against everyone who tested positive for the disease. In other words, a highly sensitive test is one that correctly identifies patients with the disease. It’s extremely rare that any clinical trial is 100% sensitive. A test with 90% sensitivity will determine 90% of patients who have the disease but will miss 10% of the patients who have the disease.

**Why does the formula use False Negatives?**
+ If an observation is a False Negative, the observation was predicted as False when it was True. Hence, we to find the total number of True Positives.

A highly sensitive test can be useful for ruling out a disease if a person has a negative result. For example, a negative outcome probably means the person does not have cervical cancer. The acronym widely used is SnNout (high Sensitivity, Negative outcome = rule out).

If the test is highly sensitive and the test result is negative, you can be nearly confident that they don’t have a disease. If you look at the formula, and there's a high sensitivity, the numerator and denominator must be around the same. Thus, the FN in the denominator must be reasonably small.


### Specific Test
+ Formula: True negative rate (or specificity): TNR=TN/(FP+TN)
+ 100/(100+5) = 95.24%

The specificity of a test (also called  True Negative Rate) is the proportion of people without the disease who will have a negative result. In other words, the specificity of a test refers to how well a test identifies patients who does not have a disease. A test that has 100% specificity will identify 100% of patients who do not have the disease. A test that is 90% specific will identify 90% of patients who do not have the disease and 10% of patients who do not have the disease.

**Why does the formula use False Positives?**
+ If an observation is a False Positive, the observation was predicted as True when it was False. Hence, we to find the total number of True Negatives.

A high specificity rate can be most useful when the result is positive. A highly specific test can be helpful in ruling in patients who have a particular disease.

In other terms, if the test result for a highly specific test is positive, you can be nearly sure that they have the disease. Similar to the concept of True Positive Rate. The inverse, 1-True Negative Rate, is FP/(FP+TN). If you look at the formula, and there's a high sensitivity, the numerator and denominator must be around the same. Thus, the FP in the denominator must be reasonably small.

### Extras
**Classifier Accuracy**
+ Formula: (true positives + true negatives) / (total examples)
+ This is a good way of measuring UNLESS the data is skewed into one direction
+ If the data is skewed, we don't really know if the model is good or we simply are predicting the model to be more like the skewed data


#### Sources:
+ [Blog Post](https://stats.stackexchange.com/questions/61829/relation-between-true-positive-false-positive-false-negative-and-true-negative)
