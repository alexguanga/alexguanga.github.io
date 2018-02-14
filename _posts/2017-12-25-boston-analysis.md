---
layout: post
title: 'Boston Analysis'
date: '2017-12-25 0:00:00 -0400'
categories: projects
description: 'The project is used to analyze the prices of houses in Boston. 
The model uses will be used is a linear regression and decision tree regressors. 
Provides an insight on how decision tree are used for continuous output and not just a classification problem.'
tags: learning, projects
permalink: bostonanalysis.html
---


# Predicting Boston Housing Prices

This project was worked through the Udemy: Data Analysis Course

In this project, we will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model will be trained to predict the prices of the homes in Boston. 

The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts.



```python
# IMPORTS

# Staple Inputs
import numpy as np
import pandas as pd
from pandas import DataFrame

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# Dataset
from sklearn.datasets import load_boston

```


```python
# Loading the Boston dataset
data_boston = load_boston()
```


```python
print(data_boston.DESCR)
```

    Boston House Prices dataset
    ===========================
    
    Notes
    ------
    Data Set Characteristics:  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive
        
        :Median Value (attribute 14) is usually the target
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    http://archive.ics.uci.edu/ml/datasets/Housing
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    **References**
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
       - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
    



```python
boston_df = DataFrame(data_boston.data)
boston_df.columns = data_boston.feature_names

# Complete df
boston_df['PRICE'] = data_boston.target

# Separating the df into their respective df
X_fts = boston_df.drop('PRICE',axis=1)
y_price =  boston_df['PRICE']
```


```python
boston_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



---

## Data Exploration


```python
boston_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.593761</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.596783</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.647423</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
boston_df.dtypes
```




    CRIM       float64
    ZN         float64
    INDUS      float64
    CHAS       float64
    NOX        float64
    RM         float64
    AGE        float64
    DIS        float64
    RAD        float64
    TAX        float64
    PTRATIO    float64
    B          float64
    LSTAT      float64
    PRICE      float64
    dtype: object




```python
# Using the apply to function to get the total amount of unique observations
boston_df.apply(lambda x: len(x.unique()))
```




    CRIM       504
    ZN          26
    INDUS       76
    CHAS         2
    NOX         81
    RM         446
    AGE        356
    DIS        412
    RAD          9
    TAX         66
    PTRATIO     46
    B          357
    LSTAT      455
    PRICE      229
    dtype: int64



- ** Contains a lot of continous variables**


```python
# Using the apply function, for every column, we find the total amount of NULL/NA values
boston_df.apply(lambda x: sum(x.isnull()))
```




    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    PRICE      0
    dtype: int64



- **That's a great thing, there is no NaN values**


```python
## Indepth look at price

min_price = np.min(y_price)
max_price = np.max(y_price)
mean_price = np.mean(y_price)
median_price = np.median(y_price)
std_price = np.std(y_price)

print("Descriptive Statistics\n")
print("The maximum price for the PRICE is {0:.2f}".format(max_price*1000))
print("The minimum price for the PRICE is {0:.2f}".format(min_price*1000))
print("The mean price for the PRICE is {0:.2f}".format(mean_price*1000))
print("The median price for the PRICE is {0:.2f}".format(median_price*1000))
print("The std fo the price for the PRICE is {0:.2f}".format(std_price*1000))

```

    Descriptive Statistics
    
    The maximum price for the PRICE is 50000.00
    The minimum price for the PRICE is 5000.00
    The mean price for the PRICE is 22532.81
    The median price for the PRICE is 21200.00
    The std fo the price for the PRICE is 9188.01


---

## Data Visualization


```python
all_cols = boston_df.columns
```


```python
# Histogram of prices (this is the target of our dataset)
plt.hist(y_price,bins=60)

plt.xlabel('Price in $1000s')
plt.ylabel('Number of houses')
plt.show()
```


![png](output_18_0.png)



```python
# creating a visual of the correlation of all the dependent variables
for i, cols in enumerate(all_cols):    
    sns.lmplot(cols, "PRICE",data = boston_df)
    plt.show()


```


![png](output_19_0.png)



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)



![png](output_19_7.png)



![png](output_19_8.png)



![png](output_19_9.png)



![png](output_19_10.png)



![png](output_19_11.png)



![png](output_19_12.png)



![png](output_19_13.png)



```python
# Compute the correlation matrix
corr = boston_df.corr()


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, annot=True, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x120ed1320>




![png](output_20_1.png)



```python
corr_matrix = boston_df.corr()
corr_matrix['PRICE'].sort_values(ascending=False)
```




    PRICE      1.000000
    RM         0.695360
    ZN         0.360445
    B          0.333461
    DIS        0.249929
    CHAS       0.175260
    AGE       -0.376955
    RAD       -0.381626
    CRIM      -0.385832
    NOX       -0.427321
    TAX       -0.468536
    INDUS     -0.483725
    PTRATIO   -0.507787
    LSTAT     -0.737663
    Name: PRICE, dtype: float64



---

## Feature Selection


```python
import statsmodels.api as sm
from scipy import stats
from collections import defaultdict

global dict_adjus_R
dict_adjus_R = defaultdict(list)

```


```python
def HighestPvalue(model, threshold):
    highest_pvalue = 0
    
    for index, current_pvalue in model.pvalues.items():
        if current_pvalue > highest_pvalue:
            highest_pvalue = current_pvalue
            highest_index = index
            
    if highest_pvalue > threshold: return highest_index
    else: return True
```


```python
def CreateLinearReg(x, y):
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2
```


```python
def BackwardElimination(Xs, y, stats_signf):
    model_info = CreateLinearReg(Xs, y)
    p_results = HighestPvalue(model_info, stats_signf)

    dict_adjus_R[len(Xs.columns)].append([Xs.columns, model_info.rsquared_adj])
    
    if p_results is True: return model_info

    else:
        Xs.drop(p_results, axis=1, inplace=True)
        BackwardElimination(Xs, y, stats_signf)

```


```python
# Statistical sigficance we would like to uses
stats_signf = 0.05
final_model = BackwardElimination(X_fts, y_price, stats_signf)
final_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>PRICE</td>      <th>  R-squared:         </th> <td>   0.741</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.735</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   128.2</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 10 Jan 2018</td> <th>  Prob (F-statistic):</th> <td>5.74e-137</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:58:52</td>     <th>  Log-Likelihood:    </th> <td> -1498.9</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3022.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   494</td>      <th>  BIC:               </th> <td>   3073.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>   36.3694</td> <td>    5.069</td> <td>    7.176</td> <td> 0.000</td> <td>   26.411</td> <td>   46.328</td>
</tr>
<tr>
  <th>CRIM</th>    <td>   -0.1076</td> <td>    0.033</td> <td>   -3.296</td> <td> 0.001</td> <td>   -0.172</td> <td>   -0.043</td>
</tr>
<tr>
  <th>ZN</th>      <td>    0.0458</td> <td>    0.014</td> <td>    3.387</td> <td> 0.001</td> <td>    0.019</td> <td>    0.072</td>
</tr>
<tr>
  <th>CHAS</th>    <td>    2.7212</td> <td>    0.854</td> <td>    3.185</td> <td> 0.002</td> <td>    1.043</td> <td>    4.400</td>
</tr>
<tr>
  <th>NOX</th>     <td>  -17.3956</td> <td>    3.536</td> <td>   -4.920</td> <td> 0.000</td> <td>  -24.343</td> <td>  -10.448</td>
</tr>
<tr>
  <th>RM</th>      <td>    3.7966</td> <td>    0.406</td> <td>    9.343</td> <td> 0.000</td> <td>    2.998</td> <td>    4.595</td>
</tr>
<tr>
  <th>DIS</th>     <td>   -1.4934</td> <td>    0.186</td> <td>   -8.039</td> <td> 0.000</td> <td>   -1.858</td> <td>   -1.128</td>
</tr>
<tr>
  <th>RAD</th>     <td>    0.2991</td> <td>    0.063</td> <td>    4.719</td> <td> 0.000</td> <td>    0.175</td> <td>    0.424</td>
</tr>
<tr>
  <th>TAX</th>     <td>   -0.0118</td> <td>    0.003</td> <td>   -3.488</td> <td> 0.001</td> <td>   -0.018</td> <td>   -0.005</td>
</tr>
<tr>
  <th>PTRATIO</th> <td>   -0.9471</td> <td>    0.129</td> <td>   -7.337</td> <td> 0.000</td> <td>   -1.201</td> <td>   -0.693</td>
</tr>
<tr>
  <th>B</th>       <td>    0.0094</td> <td>    0.003</td> <td>    3.508</td> <td> 0.000</td> <td>    0.004</td> <td>    0.015</td>
</tr>
<tr>
  <th>LSTAT</th>   <td>   -0.5232</td> <td>    0.047</td> <td>  -11.037</td> <td> 0.000</td> <td>   -0.616</td> <td>   -0.430</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>178.444</td> <th>  Durbin-Watson:     </th> <td>   1.078</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 786.944</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.524</td>  <th>  Prob(JB):          </th> <td>1.31e-171</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.295</td>  <th>  Cond. No.          </th> <td>1.47e+04</td> 
</tr>
</table>




```python
# Dictionary with some of their best models. 
# This was done to look at a group of variable and there adjusted R_Squared
dict_adjus_R
```




    defaultdict(list,
                {11: [[Index(['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                          'LSTAT'],
                         dtype='object'),
                   0.73476802182854828],
                  [Index(['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                          'LSTAT'],
                         dtype='object'), 0.73476802182854828],
                  [Index(['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                          'LSTAT'],
                         dtype='object'), 0.73476802182854828]],
                 12: [[Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX',
                          'PTRATIO', 'B', 'LSTAT'],
                         dtype='object'), 0.73429218983604261]],
                 13: [[Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                          'PTRATIO', 'B', 'LSTAT'],
                         dtype='object'), 0.7337538824121872]]})




```python
# Based on the analysis using backward elimination, I will look into using these variables
X_fts_1 = X_fts[['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 
                        'RAD', 'TAX', 'PTRATIO', 'B','LSTAT']]
```


```python
X_fts_1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



___

## Data Modeling

### Comparison
Similarities: Both MAE and RMSE express average model prediction error in units of the variable of interest. Both metrics can range from 0 to ∞ and are indifferent to the direction of errors. They are negatively-oriented scores, which means lower values are better.

Differences: Taking the square root of the average squared errors has some interesting implications for RMSE. Since the errors are squared before they are averaged, the RMSE gives a relatively high weight to large errors. This means the RMSE should be more useful when large errors are particularly undesirable. 

### Implementation: Define a Performance Metric
We will calculate the coefficient of determination, ${R}^2$, to quantify our model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions.

The values for ${R}^2$ range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the target variable. 

- ${R}^2$ of 0 is no better than a model that always predicts the mean of the target variable
- ${R}^2$ of 1 perfectly predicts the target variable
- Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the features.


```python
from sklearn.metrics import r2_score

def Score_R2(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score
```

### Training/Testing the Dataset

In a dataset a training set is implemented to build up a model, while a validation set is used to validate the model built. Data points in the training set are excluded from the validation set. The correct way to pick out samples from your dataset to be part either the training or validation (also called test) set is randomly.


```python
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
```


```python
from sklearn.model_selection import train_test_split

# Splitting the dataset to understand how the model perform on a simple split
X_train, X_test, y_train, y_test = train_test_split(X_fts_1, y_price, test_size=0.2)
model = linreg.fit(X_train, y_train)
predictions = linreg.predict(X_test)

```


```python
# Visualizing the performance of the dataset

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
```


![png](output_40_0.png)



```python
model_score = model.score(X_test, y_test)
print ("Model R_Square Performance:, {0:.2f}".format(model_score))

```

    Model R_Square Performance:, 0.72


### Cross Validation

In a dataset a training set is implemented to build up a model, while a validation set is used to validate the model built. Data points in the training set are excluded from the validation set. The correct way to pick out samples from your dataset to be part either the training or validation (also called test) set is randomly.


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

seed = 7
splits = 10
kfold = KFold(n_splits=splits, random_state=seed)

```


```python
scoring = 'neg_mean_squared_error'
results = cross_val_score(linreg, X_fts_1, y_price, cv=kfold, scoring=scoring)

all_RMSE = [np.sqrt(np.abs(result)) for result in results]
RMSE = np.sqrt(np.abs(results.mean()))
RMSE_std = np.sqrt(results.std())

print("For the RMSE, the mean is {0:.3f} and the std. deviation is {1:.3f}\n".format(RMSE, RMSE_std))
print("The RMSE for all the CV are {0}".format('\n'.join(str(r) for r in all_RMSE)))

```

    For the RMSE, the mean is 5.723 and the std. deviation is 6.451
    
    The RMSE for all the CV are 3.0325086999
    3.7300523343
    3.49240202316
    5.93185975761
    5.45459716069
    4.40422039266
    3.14890618193
    12.4212881393
    5.77112150767
    3.22201336351



```python
scoring = 'r2'

results = cross_val_score(linreg, X_fts_1, y_price, cv=kfold, scoring=scoring)
print("For the R Squared, the mean is {0:.3f} and the std. deviation is {1:.3f}\n".format(results.mean(), results.std()))
print("The R Squared for all the CV are \n{0}".format('\n'.join(str(r) for r in results)))

```

    For the R Squared, the mean is 0.247 and the std. deviation is 0.543
    
    The R Squared for all the CV are 
    0.736364963982
    0.481934238314
    -0.738769807462
    0.641343334412
    0.577913070194
    0.742233036329
    0.380262591077
    -0.0347512291036
    -0.767164243387
    0.449640823275


#### Decision Tree Regressors


```python
import visuals as vs
```


```python
# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(X_fts_1, y_price)
```


![png](output_48_0.png)


** RESULTS**

**Depth of 1:**
- This is a high bias sceanario because the score is pretty low. Hence, the model is underfitting.
- The testing score (green line) increases with the number of observations.
- The testing score only increases to approximately 0.4, a low score.
- The training score decreases to a very low score of approximately 0.4.
- This indicates how the model does not seem to fit the data well.
- **Consequently, having more training points would not benefit the model as the model is underfitting the dataset. Instead, one should increase the model complexity to better fit the dataset.**


**Depth of 3:**
- Ideal sceanrio.
- The testing score increased, but has hit a plateau at a good score (~0.7)
- The model does a good job in generalizing the data
- The training score decreases to a very low score of approximately (~0.8)
- There seems to be no high bias or high variance problem.
- **Having more training points might benefit the model as the model.**


**Depth of 6:**
- Slight high variance problem
- The training score seems to be a bit good, thus indicating a high variance problem where the model is picking up a lot of the noise
- It is overfitting
- The testing score increased, but has hit a plateau at a good score (~0.8)
- The training score decreases to a very low score of approximately (~1.0)
- **Having more training points might benefit the model as the model.**


**Depth of 10:**
- High variance problem
- The training score seems to be overfitting, thus indicating a high variance problem where the model is picking up a lot of the noise
- It is overfitting
- The testing score increased, but has hit a plateau at a good score (~0.8)
- The training score decreases to a very low score of approximately (~0.9)
- **Try smaller sets of features (bc you are overfitting).**




```python
vs.ModelComplexity(X_train, y_train)
```


![png](output_50_0.png)


**RESULTS**

The ideal depth should be 3 or 4. Of course, if its 4, it would require more computational power but it can indicaite a better model. A maximum depth of 3 also looks great to use!

### Grid Search

Grid searches specific  parameters, and the possible values of those parameters. The grid search then returns the best parameter values for our model, after fitting the supplied data. This takes out the guess-work involved in seeking out the opitimal paramter values for a classifier.

Although we will be using GridSearchCV, it may be computationally expensive for a bigger dataset.
There are other techniques that could be used for hyperparameter optimization in order to save time like RandomizedSearchCV, in this case instead of exploring the whole parameter space just a fixed number of parameter settings is sampled from the specified distributions.



```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Using the variables from the grid search
def GridSearch_DecReg(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid
```


```python
from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


def VisualizeDecisionTree(parameters, x, y):
    
    params = parameters.best_estimator_.get_params()

        
    dt_model = DecisionTreeRegressor(**params)
    dt_fit = dt_model.fit(x, y)
    ####
    
    dot_data = StringIO()
    export_graphviz(dt_fit, out_file=dot_data, special_characters=True, 
                     filled=True, rounded=True, feature_names=x.columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())
 
```


```python
grid_p = GridSearch_DecReg(X_fts_1, y_price)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {0} for the optimal model.".format(grid_p.best_estimator_.get_params()['max_depth']))

```

    Parameter 'max_depth' is 5 for the optimal model.



```python
# Checking the scores
# print(grid_p.cv_results_)
```


```python
#Creating the decision tree for the specific 
X_train, X_test, y_train, y_test = train_test_split(X_fts_1, y_price, test_size=0.2)

params = grid_p.best_estimator_.get_params()

dt_model = DecisionTreeRegressor(**params)
dt_fit = dt_model.fit(X_train, y_train)

dt_scores = cross_val_score(dt_fit, X_train, y_train, cv = 10)

r2_sqr_ytest = r2_score(y_test, grid_p.best_estimator_.predict(X_test))
score_ytest = dt_fit.score(X_test, y_test)

print("""R Squared using the predicted model using the gridCV parameters
      for the y test is {0:.2f}""".format(r2_sqr_ytest))

print("The score fitting for the testing set is {0:.2f}".format(score_ytest))
print("Mean cross validation score: {0:.2f}".format(np.mean(dt_scores)))
```

    R Squared using the predicted model using the gridCV parameters
          for the y test is 0.96
    The score fitting for the testing set is 0.74
    Mean cross validation score: 0.64



```python
VisualizeDecisionTree(grid_p, X_train, y_train)
```




![png](output_58_0.png)



** Quite intersting. Variables used were LSTAT, RM, DIS, CRIM, DIS, NOX, PTRATIO, TAX**


```python
# New set of variables
X_fts_2 = X_fts_1[['LSTAT','RM','DIS','CRIM','DIS']]
```


```python
grid_p_2 = GridSearch_DecReg(X_fts_2, y_price)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {0} for the optimal model.".format(grid_p_2.best_estimator_.get_params()['max_depth']))

```

    Parameter 'max_depth' is 5 for the optimal model.



```python
#Creating the decision tree for the specific 
X_train, X_test, y_train, y_test = train_test_split(X_fts_2, y_price, test_size=0.2)

params = grid_p_2.best_estimator_.get_params()

dt_model = DecisionTreeRegressor(**params)
dt_fit = dt_model.fit(X_train, y_train)

dt_scores = cross_val_score(dt_fit, X_train, y_train, cv = 10)

r2_sqr_ytest = r2_score(y_test, grid_p_2.best_estimator_.predict(X_test))
score_ytest = dt_fit.score(X_test, y_test)

print("""R Squared using the predicted model using the gridCV parameters
      for the y test is {0:.2f}""".format(r2_sqr_ytest))

print("The score fitting for the testing set is {0:.2f}".format(score_ytest))
print("Mean cross validation score: {0:.2f}".format(np.mean(dt_scores)))
```

    R Squared using the predicted model using the gridCV parameters
          for the y test is 0.90
    The score fitting for the testing set is 0.86
    Mean cross validation score: 0.65



```python
# Will redo the process with the top 5 variables!
vs.ModelLearning(X_fts_2, y_price)
```


![png](output_63_0.png)



```python
VisualizeDecisionTree(grid_p, X_train, y_train)
```




![png](output_64_0.png)



### Will use the first model for predictions


```python
from collections import defaultdict

# Let perform for all of them with differernt ranges
ranges = [0.025, 0.05, 0.1]
var_d = defaultdict(list)

```


```python
from collections import defaultdict

client_data = {}

for val in boston_df.columns:
    min_val = np.min(boston_df[val])
    max_val = np.max(boston_df[val])
    sampl = np.random.uniform(low=min_val, high=max_val, size=(10,))
    client_data["{0}".format(val)] = sampl

```


```python
from pandas import DataFrame
client_df = DataFrame.from_dict(client_data)
client_df = client_df[['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 
           'RAD', 'TAX', 'PTRATIO', 'B','LSTAT']]
```


```python
for i, price in enumerate(grid_p.best_estimator_.predict(client_df)):
    print ("Predicted selling price for Client {0}'s home: ${1:,.2f}".format(i+1, price*1000))
```

    Predicted selling price for Client 1's home: $20,967.76
    Predicted selling price for Client 2's home: $14,410.00
    Predicted selling price for Client 3's home: $21,900.00
    Predicted selling price for Client 4's home: $9,810.87
    Predicted selling price for Client 5's home: $14,410.00
    Predicted selling price for Client 6's home: $9,810.87
    Predicted selling price for Client 7's home: $15,539.29
    Predicted selling price for Client 8's home: $26,168.42
    Predicted selling price for Client 9's home: $15,539.29
    Predicted selling price for Client 10's home: $15,539.29


### Results
- Min Price: $5,000.00

- Max Price: $50,000.00

- Std Deviation: $9188.01

- Median Price: $21,200.00

- Mean Price: $22532.81



```python
import matplotlib.pyplot as plt
plt.hist(y_price, bins = 20)
for price in grid_p.best_estimator_.predict(client_df):
    plt.axvline(price, lw = 5, c = 'r')
```


![png](output_71_0.png)


** MOST of the data does fit within the distribution**


```python

# Using the variables from the grid search
def Temp_GridSearch(X, y):
    cv_sets = ShuffleSplit(X.shape[0], test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth': range(1,11)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_

vs.PredictTrials(X_fts_1, y_price, Temp_GridSearch, client_df.as_matrix())

```

    Trial 1: $20.34
    Trial 2: $20.50
    Trial 3: $10.90
    Trial 4: $19.93
    Trial 5: $20.34
    Trial 6: $20.67
    Trial 7: $21.05
    Trial 8: $20.58
    Trial 9: $20.89
    Trial 10: $21.77
    
    Range in prices: $10.87


