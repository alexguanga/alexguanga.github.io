---
layout: post
title: 'Linear Regression 101'
date: 'December 3rd, 2017'
categories: learning
description: 'In the simplest form, it is a line that passes, or plane, or any higher dimension figure through the points that minimizes the least squares. While this might sound abstract or weird, I will try to clarify this concept by the end of this post.'
tags: learning, articles, datascience, statistics
permalink: LinearRegression.html
---


# What is a Linear Regression?
In the purest form, it is a line that passes, or plane, or any higher dimension figure through the points that minimizes the least squares. While this might sound abstract or weird, I will try to clarify this concept by the end of this post.

Let’s think about in the mathematical equation, y = mx+b.

Let’s say we are measuring the input (hours of sleep) and calculating the output (how well you do on exams).

### Background
We get a large classroom. We ask students the number of hours they sleep and their exam grade. There are a lot of things we can measure for, but the first thing we should do is figure out the band m in the y=mx+b formula. Thus, we have new observations, like a new student coming into the classroom, we can infer it how well the student will do on the exam given their hours of sleep.

### Formula?
I know what you are probably thinking. How can we predict the values of b and m? Well, it’s through trial and error. You won’t have to do much, but whatever programming language you are using, R or Python will calculate the **Gradient Descent.**

+ The Gradient Descent finds the optimal line (in a simple regression). We will begin with an estimated value of parameters, check its prediction, compare it to the actual observations and see how far the estimations were. For all the benefits, we subtract the predicted value minus the real value. Then, we then square the result. To avoid the answer being 0. Then, add all the effects and divide it by the total number of observations. Some divide it by 2*number of observations. To minimize the result, we use partial derivatives and move towards where it’s changing the fastest.

+ If you are standing on the top a cliff, the gradient descent is telling you which way to walk towards the bottom of the cliff. The derivatives tell us that the most significant change at the given point.

### Now what?
Thus, once we find the formula that minimizes the difference between the predicted value and the observations, then we have the line of best fit. Now if we wanted to predict a new person’s test score, we can check for their hours of sleep to get so.

### That's it?
Well no. We cannot be sure that this is the best line. For example, a linear regression might not be the same model for data. It might be a polynomial to the 2nd power. How do we check it?

SER: Standard Error of the Regression or RSE: Residual Standard Error
+ Roughly speaking, it is the average amount that the response will deviate from the actual regression line. In essence, it’s adding a number value to the difference between all the error terms of each observation. The lower the number, the better the model is.

R Squared
+ Explained variation / Total variation, R-squared is always between 0 and 100%.
+ Before I go into depth, I think we need to understand some concepts beforehand.

*SSR (Sum of Squared due to Regression):* The difference between the regression and the mean.

*SSE (Sum of Squared due to Error):* The difference between the actual observation and the expected value due to the regression.

*SST:* The total sum (SSR + SSE)

+ Thus if we check the explained variation from the mean, which is expected, compared to the total change, this gives us an R Squared. If the number is close 100%, then the linear fit does a great job at predicting the model. The lower the value, the more significant the difference due to unexpected information.
