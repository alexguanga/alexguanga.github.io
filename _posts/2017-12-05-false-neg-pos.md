---
layout: post
title: 'False Positives & False Negatives'
date: 'December 5th, 2017'
categories: learning
description: 'Information on false positives and false negatives with a conceptual framework.'
tags: learning, articles, datascience, statistics
permalink: FPositiveFNegative.html
---

# Intuition

Storytelling helps convey the message better.

This concept is relatively straightforward, but the terms can sometimes be confused. Pretend you are a teacher who is tired of grading test. You have to grade plenty of tests. To save time, you decide to automate the process. The grading system is binary: Pass or Fail. You find a software developer who can create a program that will automate grading the tests. You're oblivious but take a chance anyway.

To check how well the program does, you grade a batch of exams. You then use the predictive model to grade the same batch of exams. You compare both grades to access how well the predictive model perform.

1. *True Positive*: The model says that the student passed and your analysis concurs with the model.
2. *True Negatives*: The model says that the student failed the test and your analysis concurs with the model.
3. *False Positive*: The model says that the student passed the test but your analysis failed the student.
4. *False Negatives*: The model says that the student failed the test but your analysis passed the student.

### False Positives
*This is the same as a Type 1 Error*
In our example, a false positive is someone failing the exam but the model assigning a passing grade. The severity of a false positive can be worse in different scenarios.
1. For example, our task could be to analyze whether a medical patient has a particular disease. A test is conducted to detect melanoma, a type of skin cancer. If the test is a False Positive, then the person will be labeled as sick. Then, the patient might spend a lot of money even though he never had the disease.
2. Another example could be dealing with labeling email spam or not. Machine learning has heavily used to use the content of the email to check if it is a spam or not. Imagine an email gets marked as "SPAM" even though it is an important email. Thus, the consequences can be more significant in this scenario.

### False Negatives
*This is the same as a Type 2 Error*
In our example, a false negative would be if the model predicts that a student fails while he passed it. There are also many real-life examples of this.
1. For example, if a model has to predict if a person is guilty or not, and the model predicts that the person is not liable when in fact he is, could be dangerous.
2. Another example would be claiming that a female is not pregnant, but she is.

### False Negatives VS. False Positives
We cannot aim to minimize both. This would be ideal but its impossible. I'll illustrate this using the exam example.
+ **Say we want false positives to be nonexistent:** This can be done by stating that everyone failed the exam. Our analysis says that we have no person who passed the exam. Hence, there is no way we could have falsely predicted someone to be a positive outcome (or 1) if all our predictions are negative (or 0).

+ **Say we want false negatives to be nonexistent:** This can be done by stating that everyone passed the exam. Our analysis says that we have no person who failed the exam. Hence, there is no way we could have falsely predicted someone to be a negative outcome (or 0) if all our predictions are positive (or 1).
