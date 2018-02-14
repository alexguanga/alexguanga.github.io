---
layout: post
title: 'Stock Analysis'
date: '2017-12-25 0:00:00 -0400'
categories: projects
description: 'We will be answering the following questions along the way

1. What was the change in price of the stock over time?
2. What was the daily return of the stock on average?
3. What was the moving average of the various stocks?
4. What was the correlation between different stocks closing prices?
4. What was the correlation between different stocks daily returns?
5. How much value do we put at risk by investing in a particular stock?
6. How can we attempt to predict future stock behavior?'
tags: learning, projects
permalink: stockanalysis.html
---



# Stock Market Analysis

Welcome to your second data project! In this portfolio project we will be looking at data from the stock market, particularly some technology stocks. We will learn how to use pandas to get stock information, visualize different aspects of it, and finally we will look at a few ways of analyzing the risk of a stock, based on its previous performance history. We will also be predicting future stock prices through a Monte Carlo method!

We'll be answering the following questions along the way:

1. What was the change in price of the stock over time?
2. What was the daily return of the stock on average?
3. What was the moving average of the various stocks?
4. What was the correlation between different stocks' closing prices?
4. What was the correlation between different stocks' daily returns?
5. How much value do we put at risk by investing in a particular stock?
6. How can we attempt to predict future stock behavior?



```python
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline
```


```python
import pandas_datareader as pdr
from pandas_datareader import data, wb
```


```python
from datetime import datetime
```


```python
# list of each tech stocks

tech_list = ['AAPL','FB','AMZN','GOOG']
```


```python
# making the appropriate timeframe
end = datetime.now()
start = datetime(end.year-1, end.month, end.day)
```


```python
# Quick ex of all the stocks

temp = pdr.get_data_yahoo('FB')
temp.reset_index(level=0, inplace=True) ## Changing the index value

temp.head()
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-05-18</td>
      <td>42.049999</td>
      <td>45.000000</td>
      <td>38.000000</td>
      <td>38.230000</td>
      <td>38.230000</td>
      <td>573576400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-05-21</td>
      <td>36.529999</td>
      <td>36.660000</td>
      <td>33.000000</td>
      <td>34.029999</td>
      <td>34.029999</td>
      <td>168192700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-05-22</td>
      <td>32.610001</td>
      <td>33.590000</td>
      <td>30.940001</td>
      <td>31.000000</td>
      <td>31.000000</td>
      <td>101786600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-05-23</td>
      <td>31.370001</td>
      <td>32.500000</td>
      <td>31.360001</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>73600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-05-24</td>
      <td>32.950001</td>
      <td>33.209999</td>
      <td>31.770000</td>
      <td>33.029999</td>
      <td>33.029999</td>
      <td>50237200</td>
    </tr>
  </tbody>
</table>
</div>




```python
for stock in tech_list:
    temp = pdr.get_data_yahoo(stock)
    temp.reset_index(level=0, inplace=True) ## Changing the index value
    
    globals()[stock] = temp[temp['Date'] >= start]

```


```python
# Resetting all index 
AAPL.reset_index(inplace=True)
FB.reset_index(inplace=True)
AMZN.reset_index(inplace=True)
GOOG.reset_index(inplace=True)
```


```python
# del the extra cols.
del AAPL['index']
del FB['index']
del AMZN['index']
del GOOG['index']
```


```python
GOOG.head()
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
      <th>level_0</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-01-09</td>
      <td>806.400024</td>
      <td>809.966003</td>
      <td>802.830017</td>
      <td>806.650024</td>
      <td>806.650024</td>
      <td>1272400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2017-01-10</td>
      <td>807.859985</td>
      <td>809.130005</td>
      <td>803.510010</td>
      <td>804.789978</td>
      <td>804.789978</td>
      <td>1176800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2017-01-11</td>
      <td>805.000000</td>
      <td>808.150024</td>
      <td>801.369995</td>
      <td>807.909973</td>
      <td>807.909973</td>
      <td>1065900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2017-01-12</td>
      <td>807.140015</td>
      <td>807.390015</td>
      <td>799.169983</td>
      <td>806.359985</td>
      <td>806.359985</td>
      <td>1353100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2017-01-13</td>
      <td>807.479980</td>
      <td>811.223999</td>
      <td>806.690002</td>
      <td>807.880005</td>
      <td>807.880005</td>
      <td>1099200</td>
    </tr>
  </tbody>
</table>
</div>




```python
GOOG.describe()
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
      <th>level_0</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>253.000000</td>
      <td>253.000000</td>
      <td>253.000000</td>
      <td>253.000000</td>
      <td>253.000000</td>
      <td>253.000000</td>
      <td>2.530000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>126.000000</td>
      <td>927.111421</td>
      <td>932.969089</td>
      <td>921.341079</td>
      <td>927.840138</td>
      <td>927.840138</td>
      <td>1.467822e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>73.179004</td>
      <td>79.020587</td>
      <td>80.376678</td>
      <td>78.353580</td>
      <td>79.435737</td>
      <td>79.435737</td>
      <td>6.384932e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>793.799988</td>
      <td>801.190002</td>
      <td>790.520020</td>
      <td>795.695007</td>
      <td>795.695007</td>
      <td>5.370000e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>63.000000</td>
      <td>842.880005</td>
      <td>844.909973</td>
      <td>839.320007</td>
      <td>841.650024</td>
      <td>841.650024</td>
      <td>1.086500e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>126.000000</td>
      <td>931.469971</td>
      <td>936.530029</td>
      <td>924.590027</td>
      <td>930.599976</td>
      <td>930.599976</td>
      <td>1.279500e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>189.000000</td>
      <td>980.000000</td>
      <td>985.424988</td>
      <td>972.200012</td>
      <td>977.000000</td>
      <td>977.000000</td>
      <td>1.620500e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>252.000000</td>
      <td>1109.400024</td>
      <td>1111.270020</td>
      <td>1101.619995</td>
      <td>1106.939941</td>
      <td>1106.939941</td>
      <td>5.167700e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
GOOG.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 253 entries, 0 to 252
    Data columns (total 8 columns):
    level_0      253 non-null int64
    Date         253 non-null datetime64[ns]
    Open         253 non-null float64
    High         253 non-null float64
    Low          253 non-null float64
    Close        253 non-null float64
    Adj Close    253 non-null float64
    Volume       253 non-null int64
    dtypes: datetime64[ns](1), float64(5), int64(2)
    memory usage: 15.9 KB



```python
GOOG['Adj Close'].plot(legend=True,figsize = (10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1141b5908>




![png](output_14_1.png)



```python
GOOG.plot(y="Volume",x="Date",legend=True,figsize = (10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1118cfb38>




![png](output_15_1.png)


### Info on moving averages
http://www.investopedia.com/terms/m/movingaverage.asp

http://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp


```python
ma_day = [10,20,50]

for ma in ma_day:
    colname = "MA for {0} days".format(ma)
    
    # For the AAPL STOCK
    AAPL[colname] = AAPL['Adj Close'].rolling(window=ma).mean()
```


```python
AAPL.head(10)
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
      <th>level_0</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>MA for 10 days</th>
      <th>MA for 20 days</th>
      <th>MA for 50 days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-01-09</td>
      <td>117.949997</td>
      <td>119.430000</td>
      <td>117.940002</td>
      <td>118.989998</td>
      <td>117.106812</td>
      <td>33561900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2017-01-10</td>
      <td>118.769997</td>
      <td>119.379997</td>
      <td>118.300003</td>
      <td>119.110001</td>
      <td>117.224907</td>
      <td>24462100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2017-01-11</td>
      <td>118.739998</td>
      <td>119.930000</td>
      <td>118.599998</td>
      <td>119.750000</td>
      <td>117.854782</td>
      <td>27588600</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2017-01-12</td>
      <td>118.900002</td>
      <td>119.300003</td>
      <td>118.209999</td>
      <td>119.250000</td>
      <td>117.362694</td>
      <td>27086200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2017-01-13</td>
      <td>119.110001</td>
      <td>119.620003</td>
      <td>118.809998</td>
      <td>119.040001</td>
      <td>117.156021</td>
      <td>26111900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2017-01-17</td>
      <td>118.339996</td>
      <td>120.239998</td>
      <td>118.220001</td>
      <td>120.000000</td>
      <td>118.100822</td>
      <td>34439800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2017-01-18</td>
      <td>120.000000</td>
      <td>120.500000</td>
      <td>119.709999</td>
      <td>119.989998</td>
      <td>118.090981</td>
      <td>23713000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2017-01-19</td>
      <td>119.400002</td>
      <td>120.089996</td>
      <td>119.370003</td>
      <td>119.779999</td>
      <td>117.884300</td>
      <td>25597300</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>2017-01-20</td>
      <td>120.449997</td>
      <td>120.449997</td>
      <td>119.730003</td>
      <td>120.000000</td>
      <td>118.100822</td>
      <td>32597900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2017-01-23</td>
      <td>120.000000</td>
      <td>120.809998</td>
      <td>119.769997</td>
      <td>120.080002</td>
      <td>118.179558</td>
      <td>22050200</td>
      <td>117.70617</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
AAPL[["Adj Close", "MA for 10 days","MA for 20 days","MA for 50 days"]].plot(subplots=False, figsize = (10,4))
plt.show()
```


![png](output_19_0.png)



```python
AAPL['Daily Returns'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Returns'].plot(figsize=(14,6), legend=True, linestyle='--', marker='o')
plt.show()
```


![png](output_20_0.png)



```python
sns.distplot(AAPL['Daily Returns'].dropna(), bins=100, color='purple')
plt.show()
```


![png](output_21_0.png)



```python
AAPL['Daily Returns'].hist(bins=100)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1188e0f60>




![png](output_22_1.png)



```python
AAPL.head()
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
      <th>level_0</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>MA for 10 days</th>
      <th>MA for 20 days</th>
      <th>MA for 50 days</th>
      <th>Daily Returns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-01-09</td>
      <td>117.949997</td>
      <td>119.430000</td>
      <td>117.940002</td>
      <td>118.989998</td>
      <td>117.106812</td>
      <td>33561900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2017-01-10</td>
      <td>118.769997</td>
      <td>119.379997</td>
      <td>118.300003</td>
      <td>119.110001</td>
      <td>117.224907</td>
      <td>24462100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2017-01-11</td>
      <td>118.739998</td>
      <td>119.930000</td>
      <td>118.599998</td>
      <td>119.750000</td>
      <td>117.854782</td>
      <td>27588600</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005373</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2017-01-12</td>
      <td>118.900002</td>
      <td>119.300003</td>
      <td>118.209999</td>
      <td>119.250000</td>
      <td>117.362694</td>
      <td>27086200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2017-01-13</td>
      <td>119.110001</td>
      <td>119.620003</td>
      <td>118.809998</td>
      <td>119.040001</td>
      <td>117.156021</td>
      <td>26111900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001761</td>
    </tr>
  </tbody>
</table>
</div>




```python
AAPL['Adj Close'].head()
```




    0    117.106812
    1    117.224907
    2    117.854782
    3    117.362694
    4    117.156021
    Name: Adj Close, dtype: float64




```python
GOOG['Adj Close'].head()
```




    0    806.650024
    1    804.789978
    2    807.909973
    3    806.359985
    4    807.880005
    Name: Adj Close, dtype: float64




```python
FB['Adj Close'].head()
```




    0    124.900002
    1    124.349998
    2    126.089996
    3    126.620003
    4    128.339996
    Name: Adj Close, dtype: float64




```python
#closing_df = pdr(tech_list, 'yahoo', start, end)['Adj Close']
closing_df = DataFrame({
    "AAPL": AAPL['Adj Close'],
    "GOOG": GOOG['Adj Close'],
    "FB": FB['Adj Close'],
    "AMZN": AMZN['Adj Close']
    
})
closing_df.head()

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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117.106812</td>
      <td>796.919983</td>
      <td>124.900002</td>
      <td>806.650024</td>
    </tr>
    <tr>
      <th>1</th>
      <td>117.224907</td>
      <td>795.900024</td>
      <td>124.349998</td>
      <td>804.789978</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117.854782</td>
      <td>799.020020</td>
      <td>126.089996</td>
      <td>807.909973</td>
    </tr>
    <tr>
      <th>3</th>
      <td>117.362694</td>
      <td>813.640015</td>
      <td>126.620003</td>
      <td>806.359985</td>
    </tr>
    <tr>
      <th>4</th>
      <td>117.156021</td>
      <td>817.140015</td>
      <td>128.339996</td>
      <td>807.880005</td>
    </tr>
  </tbody>
</table>
</div>




```python
big4_returns = closing_df.pct_change()

big4_returns.drop(big4_returns.index[0], inplace=True)
```


```python
big4_returns.head()

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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.001008</td>
      <td>-0.001280</td>
      <td>-0.004404</td>
      <td>-0.002306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005373</td>
      <td>0.003920</td>
      <td>0.013993</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.004175</td>
      <td>0.018297</td>
      <td>0.004203</td>
      <td>-0.001919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.001761</td>
      <td>0.004302</td>
      <td>0.013584</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008064</td>
      <td>-0.009081</td>
      <td>-0.003662</td>
      <td>-0.004048</td>
    </tr>
  </tbody>
</table>
</div>




```python
for x in tech_list:
    for y in tech_list:
        if x != y:
            sns.jointplot(x, y, big4_returns)
            plt.title(('{0} and {1}').format(x,y)) # Various CI
            plt.show()

```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)



![png](output_30_3.png)



![png](output_30_4.png)



![png](output_30_5.png)



![png](output_30_6.png)



![png](output_30_7.png)



![png](output_30_8.png)



![png](output_30_9.png)



![png](output_30_10.png)



![png](output_30_11.png)



```python
from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')
```




![svg](output_31_0.svg)




```python
big4_returns.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.001008</td>
      <td>-0.001280</td>
      <td>-0.004404</td>
      <td>-0.002306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005373</td>
      <td>0.003920</td>
      <td>0.013993</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.004175</td>
      <td>0.018297</td>
      <td>0.004203</td>
      <td>-0.001919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.001761</td>
      <td>0.004302</td>
      <td>0.013584</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008064</td>
      <td>-0.009081</td>
      <td>-0.003662</td>
      <td>-0.004048</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pairplot
sns.pairplot(big4_returns.dropna())
```




    <seaborn.axisgrid.PairGrid at 0x11b5e7978>




![png](output_33_1.png)



```python
# create our pairplot

returns_fig = sns.PairGrid(big4_returns.dropna())

returns_fig.map_upper(plt.scatter,color='orange')
returns_fig.map_lower(sns.kdeplot,cmpa='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.show()
```


![png](output_34_0.png)



```python
# create our pairplot

returns_fig = sns.PairGrid(closing_df.dropna())

returns_fig.map_upper(plt.scatter,color='orange')
returns_fig.map_lower(sns.kdeplot,cmpa='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
plt.show()
```


![png](output_35_0.png)



```python
big4_returns.corr()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>1.000000</td>
      <td>0.511322</td>
      <td>0.545902</td>
      <td>0.491142</td>
    </tr>
    <tr>
      <th>AMZN</th>
      <td>0.511322</td>
      <td>1.000000</td>
      <td>0.653747</td>
      <td>0.670783</td>
    </tr>
    <tr>
      <th>FB</th>
      <td>0.545902</td>
      <td>0.653747</td>
      <td>1.000000</td>
      <td>0.711600</td>
    </tr>
    <tr>
      <th>GOOG</th>
      <td>0.491142</td>
      <td>0.670783</td>
      <td>0.711600</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute the correlation matrix
corr = big4_returns.corr()

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




    <matplotlib.axes._subplots.AxesSubplot at 0x11a9b6588>




![png](output_37_1.png)



```python
# Compute the correlation matrix
corr = closing_df.corr()

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




    <matplotlib.axes._subplots.AxesSubplot at 0x11b282400>




![png](output_38_1.png)


## Risk Analysis


```python
rets = big4_returns.dropna()
area = np.pi * 20
plt.scatter(x=rets.mean(), y=rets.std(), s=area)

plt.xlabel('Expected Return')
plt.ylabel('Risk')


# Label the scatter plots, for more info on how this is done, chekc out the link below
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))

```


![png](output_40_0.png)


## Value at Risk
We can treat value at risk as the amount of money we could expect to lose (aka putting at risk) for a given confidence interval. Theres several methods we can use for estimating a value at risk. Let's go ahead and see some of them in action.

## Value at risk using the "bootstrap" method

For this method we will calculate the empirical quantiles from a histogram of daily returns. For more information on quantiles, check out this link: http://en.wikipedia.org/wiki/Quantile


```python
# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
sns.distplot(AAPL['Daily Returns'].dropna(),bins=100,color='purple')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11aa40630>




![png](output_42_1.png)



```python
# The 0.05 empirical quantile of daily returns
rets['AAPL'].quantile(0.05)
```




    -0.014911698036799348



** The 0.05 empirical quantile of daily returns is at -0.014. That means that with 95% confidence, our worst daily loss will not exceed 1.4%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.014 * 1,000,000 = $14,000.**


```python
from collections import defaultdict

# Let perform for all of them with differernt ranges
ranges = [0.025, 0.05, 0.1]
var_d = defaultdict(list)


for t_stock in tech_list:
    for val in ranges:
        ret = rets[t_stock].quantile(val)
        var_d[t_stock].append(ret)
var_d
```




    defaultdict(list,
                {'AAPL': [-0.020318352248894993,
                  -0.014911698036799348,
                  -0.009775791014228942],
                 'AMZN': [-0.023780200527053923,
                  -0.015361122650408596,
                  -0.010369152275866923],
                 'FB': [-0.02066069191606408,
                  -0.01549650704753944,
                  -0.008454008608159902],
                 'GOOG': [-0.02331393054836877,
                  -0.014096341117248605,
                  -0.008840928304327932]})




```python
df_VaR = DataFrame(var_d)
df_VaR
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.020318</td>
      <td>-0.023780</td>
      <td>-0.020661</td>
      <td>-0.023314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.014912</td>
      <td>-0.015361</td>
      <td>-0.015497</td>
      <td>-0.014096</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.009776</td>
      <td>-0.010369</td>
      <td>-0.008454</td>
      <td>-0.008841</td>
    </tr>
  </tbody>
</table>
</div>



## Efficient Market Hypothesis

- The three versions of the efficient market hypothesis are varying degrees of the same basic theory. The weak form suggests that today’s stock prices reflect all the data of past prices and that no form of technical analysis can be effectively utilized to aid investors in making trading decisions. Advocates for the weak form efficiency theory allow that if fundamental analysis is used, undervalued and overvalued stocks can be determined, and investors can research companies' financial statements to increase their chances of making higher-than-market-average profits.

- The semi-strong form efficiency theory follows the belief that because all information that is public is used in the calculation of a stock's current price, investors cannot utilize either technical or fundamental analysis to gain higher returns in the market. Those who subscribe to this version of the theory believe that only information that is not readily available to the public can help investors boost their returns to a performance level above that of the general market. Inside trading.

- The strong form version of the efficient market hypothesis states that all information – both the information available to the public and any information not publicly known – is completely accounted for in current stock prices, and there is no type of information that can give an investor an advantage on the market. Advocates for this degree of the theory suggest that investors cannot make returns on investments that exceed normal market returns, regardless of information retrieved or research conducted.

- https://www.investopedia.com/ask/answers/032615/what-are-differences-between-weak-strong-and-semistrong-versions-efficient-market-hypothesis.asp#ixzz52OLUr7S1 


## Value at Risk using the Monte Carlo method
- Using the Monte Carlo to run many trials with random market conditions, then we'll calculate portfolio losses for each trial. After this, we'll use the aggregation of all these simulations to establish how risky the stock is.

We will use the geometric Brownian motion (GBM), which is technically known as a Markov process. This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of the efficient market hypothesis (EMH): past price information is already incorporated and the next price movement is "conditionally independent" of past price movements.

**DRIFT**: Expected Periodic Daily Rate of Return (the rate with the greatest odds of returning)

We will use the geometric Brownian motion (GBM), which is technically known as a Markov process. This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of the efficient market hypothesis (EMH): past price information is already incorporated and the next price movement is "conditionally independent" of past price movements.

This means that the past information on the price of a stock is independent of where the stock price will be in the future, basically meaning, you can't perfectly predict the future solely based on the previous price of a stock.

The equation for geometric Browninan motion is given by the following equation:
<img src="Monte_Carlo_1.png">

We can mulitply both sides by the stock price (S) to rearrange the formula and solve for the stock price.
<img src="Monte_Carlo_2.png">
- The first term is known as "drift", which is the average daily return multiplied by the change of time. 
- The second term is known as "shock", for each time period the stock will "drift" and then experience a "shock" which will randomly push the stock price up or down. 


**By simulating this series of steps of drift and shock thousands of times, we can begin to do a simulation of where we might expect the stock price to be.**


For more info on the Monte Carlo method for stocks, check out the following link: http://www.investopedia.com/articles/07/montecarlo.asp


```python
big4_returns.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.001008</td>
      <td>-0.001280</td>
      <td>-0.004404</td>
      <td>-0.002306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005373</td>
      <td>0.003920</td>
      <td>0.013993</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.004175</td>
      <td>0.018297</td>
      <td>0.004203</td>
      <td>-0.001919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.001761</td>
      <td>0.004302</td>
      <td>0.013584</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008064</td>
      <td>-0.009081</td>
      <td>-0.003662</td>
      <td>-0.004048</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Setting time horizon

days = 365
dt = 1/days
```


```python
rets.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.001008</td>
      <td>-0.001280</td>
      <td>-0.004404</td>
      <td>-0.002306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005373</td>
      <td>0.003920</td>
      <td>0.013993</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.004175</td>
      <td>0.018297</td>
      <td>0.004203</td>
      <td>-0.001919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.001761</td>
      <td>0.004302</td>
      <td>0.013584</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008064</td>
      <td>-0.009081</td>
      <td>-0.003662</td>
      <td>-0.004048</td>
    </tr>
  </tbody>
</table>
</div>




```python
def stock_monte_carlo(*args):
    start_price, days, mu, sigma = args
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1, days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        
        drift[x] = mu*dt
        
        price[x] = price[x-1] +(price[x-1] * (drift[x] + shock[x]))
    
    return price
```


```python
def get_metrics(tck, df):
    mu = df.mean()[tck]
    sigma = df.std()[tck]
    return(mu, sigma, tck)
    
```


```python
def run_simulations(*args):
    days = 365
    start_price, mu, sigma = args
    
    runs = 10000
    simulation = np.zeros(runs)

    for run in range(runs):
        simulation[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1] # do not know why days-1
    return simulation
```


```python

def plot_histogram(*args):
    simulation, start_price, tck = args
    
    q = np.percentile(simulation,1)
    plt.hist(simulation, 150)

    plt.figtext(0.62,0.8, s="Start Price: {0:.2f}".format(start_price))

    # Mean
    plt.figtext(0.62,0.7, s="Mean final price: {0:.2f}".format(simulation.mean()))

    # Variance
    plt.figtext(0.62,0.6, s="VaR(0.99): {0:.2f}".format(start_price-q))

    # Display 1 Percent Quantile
    plt.figtext(0.13,0.6,"q(0.99): {0:.2f}".format(q))

    # Plot a line at the 1% quantile result
    plt.axvline(x=q, linewidth=4, color='r')

    plt.title("Final price distribution for {0} Stock after 365 days".format(tck))


```


```python
def create_monte_chart(*args):
    start_price = args[0]
    mu, sigma, tck = args[1]
    days = 365
        
    for run in range(100):
        plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Analysis for {0}".format(tck))
    plt.show()
    plt.close()
    
```

** The Mean Final price is close to the starting price because the expected return is close to 0. The VaR means that if we run this simulation after many times, we will encounter a lost of 2.63. This is NOT a big loss (indicated by the red line) **


```python
start_price = AAPL['Adj Close'][0]
create_monte_chart(start_price, get_metrics('AAPL', rets))
```


![png](output_58_0.png)



```python
start_price = FB['Adj Close'][0]
create_monte_chart(start_price, get_metrics('FB', rets))

```


![png](output_59_0.png)



```python
start_price = AMZN['Adj Close'][0]
create_monte_chart(start_price, get_metrics('AMZN', rets))

```


![png](output_60_0.png)



```python
start_price = GOOG['Adj Close'][0]
create_monte_chart(start_price, get_metrics('GOOG', rets))

```


![png](output_61_0.png)



```python
rets.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.001008</td>
      <td>-0.001280</td>
      <td>-0.004404</td>
      <td>-0.002306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005373</td>
      <td>0.003920</td>
      <td>0.013993</td>
      <td>0.003877</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.004175</td>
      <td>0.018297</td>
      <td>0.004203</td>
      <td>-0.001919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.001761</td>
      <td>0.004302</td>
      <td>0.013584</td>
      <td>0.001885</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008064</td>
      <td>-0.009081</td>
      <td>-0.003662</td>
      <td>-0.004048</td>
    </tr>
  </tbody>
</table>
</div>




```python
start_price = FB['Adj Close'][0]

mu, sigma, tck = get_metrics('FB', rets)

simulations = run_simulations(start_price, mu, sigma)
plot_histogram(simulations, start_price, tck)
```


![png](output_63_0.png)



```python
start_price = AAPL['Adj Close'][0]

mu, sigma, tck = get_metrics('AAPL', rets)

simulations = run_simulations(start_price, mu, sigma)
plot_histogram(simulations, start_price, tck)
```


![png](output_64_0.png)



```python
start_price = AMZN['Adj Close'][0]

mu, sigma, tck = get_metrics('AMZN', rets)

simulations = run_simulations(start_price, mu, sigma)
plot_histogram(simulations, start_price, tck)
```


![png](output_65_0.png)



```python
start_price = GOOG['Adj Close'][0]

mu, sigma, tck = get_metrics('GOOG', big4_returns)

simulations = run_simulations(start_price, mu, sigma)
plot_histogram(simulations, start_price, tck)
```
