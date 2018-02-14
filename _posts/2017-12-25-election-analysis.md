---
layout: post
title: 'Election Analysis'
date: '2017-12-25 0:00:00 -0400'
categories: projects
description: 'In this project we will analyze two datasets. The first data set will be the results of political polls. We will analyze this aggregated poll data and answer some questions:

1. Who was being polled and what was their party affiliation?
2. Did the poll results favor Trump or Clinton?
3. How do undecided voters effect the poll?
4. Can we account for the undecided voters?
5. How did voter sentiment change over time?
6. Can we see an effect in the polls from the debates?'
tags: learning, projects
permalink: electionanalysis.html
---


# Election Data Analysis

In this project we will analyze two datasets. The first data set will be the results of political polls. We will analyze this aggregated poll data and answer some questions:

1. Who was being polled and what was their party affiliation?
2. Did the poll results favor Trump or Clinton?
3. How do undecided voters effect the poll?
4. Can we account for the undecided voters?
5. How did voter sentiment change over time?
6. Can we see an effect in the polls from the debates?




```python
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

%matplotlib inline

import requests # API
from io import StringIO
```

- The data for the polls will be obtained from HuffPost Pollster. You can check their website http://elections.huffingtonpost.com/pollster.


```python
# This is the url link for the poll data in csv form
url = "http://elections.huffingtonpost.com/pollster/2016-general-election-trump-vs-clinton.csv"

data = requests.get(url).text

poll_data = StringIO(data)
```


```python
poll_df = pd.read_csv(poll_data)
poll_df.head()
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
      <th>Pollster</th>
      <th>Start Date</th>
      <th>End Date</th>
      <th>Entry Date/Time (ET)</th>
      <th>Number of Observations</th>
      <th>Population</th>
      <th>Mode</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Pollster URL</th>
      <th>Source URL</th>
      <th>Partisan</th>
      <th>Affiliation</th>
      <th>Question Text</th>
      <th>Question Iteration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>940.0</td>
      <td>Likely Voters</td>
      <td>Internet</td>
      <td>41.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - Democrat</td>
      <td>Internet</td>
      <td>6.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - Republican</td>
      <td>Internet</td>
      <td>82.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - independent</td>
      <td>Internet</td>
      <td>38.0</td>
      <td>43.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IBD/TIPP</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:10:06Z</td>
      <td>1107.0</td>
      <td>Likely Voters</td>
      <td>Live Phone</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.investors.com/politics/ibd-tipp-pre...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
poll_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1522 entries, 0 to 1521
    Data columns (total 17 columns):
    Pollster                  1522 non-null object
    Start Date                1522 non-null object
    End Date                  1522 non-null object
    Entry Date/Time (ET)      1522 non-null object
    Number of Observations    1013 non-null float64
    Population                1522 non-null object
    Mode                      1522 non-null object
    Trump                     1522 non-null float64
    Clinton                   1522 non-null float64
    Other                     1098 non-null float64
    Undecided                 1460 non-null float64
    Pollster URL              1522 non-null object
    Source URL                1522 non-null object
    Partisan                  1522 non-null object
    Affiliation               1522 non-null object
    Question Text             661 non-null object
    Question Iteration        1522 non-null int64
    dtypes: float64(5), int64(1), object(11)
    memory usage: 202.2+ KB



```python
poll_df.describe()
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
      <th>Number of Observations</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Question Iteration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1013.000000</td>
      <td>1522.00000</td>
      <td>1522.000000</td>
      <td>1098.000000</td>
      <td>1460.000000</td>
      <td>1522.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1916.022705</td>
      <td>40.64389</td>
      <td>42.733903</td>
      <td>5.806011</td>
      <td>9.315068</td>
      <td>1.216820</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5050.240246</td>
      <td>23.56639</td>
      <td>25.298731</td>
      <td>5.009533</td>
      <td>6.253118</td>
      <td>0.412214</td>
    </tr>
    <tr>
      <th>min</th>
      <td>59.000000</td>
      <td>2.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>543.000000</td>
      <td>32.00000</td>
      <td>27.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>894.000000</td>
      <td>39.00000</td>
      <td>42.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1281.000000</td>
      <td>45.00000</td>
      <td>50.000000</td>
      <td>8.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70194.000000</td>
      <td>93.00000</td>
      <td>96.000000</td>
      <td>34.000000</td>
      <td>36.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
poll_df.columns
```




    Index(['Pollster', 'Start Date', 'End Date', 'Entry Date/Time (ET)',
           'Number of Observations', 'Population', 'Mode', 'Trump', 'Clinton',
           'Other', 'Undecided', 'Pollster URL', 'Source URL', 'Partisan',
           'Affiliation', 'Question Text', 'Question Iteration'],
          dtype='object')




```python
# Checking the affiliation and parisian
sns.factorplot('Affiliation', data=poll_df, kind="count")
plt.show()
```


![png](output_8_0.png)



```python
sns.factorplot('Affiliation', data=poll_df, kind='count', hue='Population')
plt.show()
```


![png](output_9_0.png)


**Results**
- Most people did not want to identify with any party
- People who weren't affilated with no party had a similar population
- Democrats and Repulicans were not outwardly backed


```python
# Averages
avg = pd.DataFrame(poll_df.mean())

avg.drop(['Question Iteration', 'Number of Observations'], axis=0, inplace=True)
avg.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Trump</th>
      <td>40.643890</td>
    </tr>
    <tr>
      <th>Clinton</th>
      <td>42.733903</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>5.806011</td>
    </tr>
    <tr>
      <th>Undecided</th>
      <td>9.315068</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Standard Devation
std = pd.DataFrame(poll_df.std())

std.drop(['Number of Observations', 'Question Iteration'], axis=0, inplace=True)
std.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Trump</th>
      <td>23.566390</td>
    </tr>
    <tr>
      <th>Clinton</th>
      <td>25.298731</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>5.009533</td>
    </tr>
    <tr>
      <th>Undecided</th>
      <td>6.253118</td>
    </tr>
  </tbody>
</table>
</div>




```python
avg.plot(yerr=std, kind='bar', legend=False) # Y error means the black line on top of the bar graph

```




    <matplotlib.axes._subplots.AxesSubplot at 0x116788588>




![png](output_13_1.png)



```python
poll_std_avg = pd.concat([avg, std], axis=1)
poll_std_avg.columns = ['Average','Std. Deviation']
poll_std_avg
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
      <th>Average</th>
      <th>Std. Deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Trump</th>
      <td>40.643890</td>
      <td>23.566390</td>
    </tr>
    <tr>
      <th>Clinton</th>
      <td>42.733903</td>
      <td>25.298731</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>5.806011</td>
      <td>5.009533</td>
    </tr>
    <tr>
      <th>Undecided</th>
      <td>9.315068</td>
      <td>6.253118</td>
    </tr>
  </tbody>
</table>
</div>



** Race is a pretty close between Trump and Clinton. Other and Undecided should be looked at more closely.**


```python
poll_df.head()
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
      <th>Pollster</th>
      <th>Start Date</th>
      <th>End Date</th>
      <th>Entry Date/Time (ET)</th>
      <th>Number of Observations</th>
      <th>Population</th>
      <th>Mode</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Pollster URL</th>
      <th>Source URL</th>
      <th>Partisan</th>
      <th>Affiliation</th>
      <th>Question Text</th>
      <th>Question Iteration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>940.0</td>
      <td>Likely Voters</td>
      <td>Internet</td>
      <td>41.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - Democrat</td>
      <td>Internet</td>
      <td>6.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - Republican</td>
      <td>Internet</td>
      <td>82.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - independent</td>
      <td>Internet</td>
      <td>38.0</td>
      <td>43.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IBD/TIPP</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:10:06Z</td>
      <td>1107.0</td>
      <td>Likely Voters</td>
      <td>Live Phone</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.investors.com/politics/ibd-tipp-pre...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A quick time-series about how people did vote for both of these candidates

end_date_df = poll_df.groupby(['End Date'], as_index=False).mean()

end_date_df.plot(x="End Date", y=['Trump','Clinton','Undecided', 'Other'], marker='o', linestyle='')
plt.show()
```


![png](output_17_0.png)



```python
# For timestamps
from datetime import datetime
```


```python
# Difference btw. both candidates
# Positive value: Clinton leads
# Negative value: Trump leads

poll_df['Difference'] = (poll_df.Clinton-poll_df.Trump)/100
poll_df.head(2)
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
      <th>Pollster</th>
      <th>Start Date</th>
      <th>End Date</th>
      <th>Entry Date/Time (ET)</th>
      <th>Number of Observations</th>
      <th>Population</th>
      <th>Mode</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Pollster URL</th>
      <th>Source URL</th>
      <th>Partisan</th>
      <th>Affiliation</th>
      <th>Question Text</th>
      <th>Question Iteration</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>940.0</td>
      <td>Likely Voters</td>
      <td>Internet</td>
      <td>41.0</td>
      <td>45.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Insights West</td>
      <td>2016-11-04</td>
      <td>2016-11-07</td>
      <td>2016-11-08T12:16:30Z</td>
      <td>NaN</td>
      <td>Likely Voters - Democrat</td>
      <td>Internet</td>
      <td>6.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>http://elections.huffingtonpost.com/pollster/p...</td>
      <td>http://www.insightswest.com/news/clinton-is-ah...</td>
      <td>Nonpartisan</td>
      <td>None</td>
      <td>As you may know, there will be a presidential ...</td>
      <td>1</td>
      <td>0.83</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Organize the data through start_date
# Looking at the mean

poll_start_df = poll_df.groupby(['Start Date'], as_index=False).mean()
poll_start_df.head()
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
      <th>Start Date</th>
      <th>Number of Observations</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Question Iteration</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-05-19</td>
      <td>1046.00</td>
      <td>34.25</td>
      <td>48.75</td>
      <td>2.5</td>
      <td>14.00</td>
      <td>1.0</td>
      <td>0.1450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-06-20</td>
      <td>420.75</td>
      <td>35.00</td>
      <td>47.25</td>
      <td>NaN</td>
      <td>17.75</td>
      <td>1.0</td>
      <td>0.1225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-06-21</td>
      <td>1005.00</td>
      <td>34.00</td>
      <td>51.00</td>
      <td>3.0</td>
      <td>12.00</td>
      <td>1.0</td>
      <td>0.1700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-06-26</td>
      <td>890.00</td>
      <td>36.75</td>
      <td>57.00</td>
      <td>6.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>0.2025</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-07-09</td>
      <td>499.25</td>
      <td>35.25</td>
      <td>49.50</td>
      <td>NaN</td>
      <td>16.00</td>
      <td>1.0</td>
      <td>0.1425</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Anything positive means that Clinton had a huge lead
# There are a couple of negative spikes indicating that Trump got a huge turnout in those dates

poll_start_df.plot(x='Start Date',y='Difference',figsize=(12,4), marker='o',linestyle='-', color='red')
plt.show()
```


![png](output_21_0.png)



```python
# Finding those big negtative spike
poll_start_df.nsmallest(10,'Difference')
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
      <th>Start Date</th>
      <th>Number of Observations</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Question Iteration</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2015-08-21</td>
      <td>3567.000000</td>
      <td>54.000000</td>
      <td>46.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.080000</td>
    </tr>
    <tr>
      <th>223</th>
      <td>2016-09-20</td>
      <td>1000.000000</td>
      <td>44.750000</td>
      <td>37.750000</td>
      <td>5.000000</td>
      <td>4.750000</td>
      <td>1.000000</td>
      <td>-0.070000</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2016-05-19</td>
      <td>1000.500000</td>
      <td>43.750000</td>
      <td>38.250000</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>-0.055000</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2016-02-11</td>
      <td>495.500000</td>
      <td>46.750000</td>
      <td>41.500000</td>
      <td>NaN</td>
      <td>12.500000</td>
      <td>1.000000</td>
      <td>-0.052500</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-09-02</td>
      <td>900.000000</td>
      <td>45.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>-0.050000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2015-10-10</td>
      <td>1004.000000</td>
      <td>45.000000</td>
      <td>40.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>-0.050000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2015-10-01</td>
      <td>838.666667</td>
      <td>46.000000</td>
      <td>41.250000</td>
      <td>NaN</td>
      <td>12.750000</td>
      <td>1.000000</td>
      <td>-0.047500</td>
    </tr>
    <tr>
      <th>122</th>
      <td>2016-05-14</td>
      <td>1146.333333</td>
      <td>41.916667</td>
      <td>37.666667</td>
      <td>6.750000</td>
      <td>9.583333</td>
      <td>1.333333</td>
      <td>-0.042500</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2015-12-03</td>
      <td>1023.500000</td>
      <td>44.750000</td>
      <td>40.500000</td>
      <td>NaN</td>
      <td>15.000000</td>
      <td>1.000000</td>
      <td>-0.042500</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2016-07-22</td>
      <td>1136.928571</td>
      <td>43.758621</td>
      <td>39.551724</td>
      <td>3.470588</td>
      <td>8.965517</td>
      <td>1.413793</td>
      <td>-0.042069</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Finding those big negtative spike
trump_dates = poll_start_df.nsmallest(5,'Difference')['Start Date']
trump_dates = list(trump_dates)
```

** Debates were September 26th, October 9, and October 19**

** Let look at the dates of the debate to see if they can provide any additional information**


```python
def timeframe(year, month):
    row = 0
    xindex = []

    for date in poll_start_df['Start Date']:
        date = datetime.strptime(date, "%Y-%m-%d")
        if ((date.year == year) and (date.month == month)):
            xindex.append(row)
            row += 1
        else:
            row += 1

    return(min(xindex), max(xindex)) # Checking the range of the index that I will be using
```


```python
# Narrow down the dates of late September and Mid October
year = 2016
month = 10

result = timeframe(year, month)

print("Max date is {0}. Min date is {1}".format(min(result), max(result)))
```

    Max date is 232. Min date is 262



```python
poll_start_df.loc[232:262].head()
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
      <th>Start Date</th>
      <th>Number of Observations</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Question Iteration</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>232</th>
      <td>2016-10-01</td>
      <td>527.500000</td>
      <td>42.250000</td>
      <td>41.750000</td>
      <td>4.750000</td>
      <td>6.750000</td>
      <td>1.0</td>
      <td>-0.005000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2016-10-02</td>
      <td>451.000000</td>
      <td>37.000000</td>
      <td>45.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>1.0</td>
      <td>0.080000</td>
    </tr>
    <tr>
      <th>234</th>
      <td>2016-10-03</td>
      <td>12226.000000</td>
      <td>42.900000</td>
      <td>43.400000</td>
      <td>4.333333</td>
      <td>4.222222</td>
      <td>1.1</td>
      <td>0.005000</td>
    </tr>
    <tr>
      <th>235</th>
      <td>2016-10-04</td>
      <td>1500.000000</td>
      <td>42.000000</td>
      <td>43.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2016-10-05</td>
      <td>827.666667</td>
      <td>39.333333</td>
      <td>46.666667</td>
      <td>3.500000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>0.073333</td>
    </tr>
  </tbody>
</table>
</div>




```python
poll_start_df.loc[232:262].tail()
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
      <th>Start Date</th>
      <th>Number of Observations</th>
      <th>Trump</th>
      <th>Clinton</th>
      <th>Other</th>
      <th>Undecided</th>
      <th>Question Iteration</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>2016-10-27</td>
      <td>1249.000000</td>
      <td>47.000000</td>
      <td>52.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>259</th>
      <td>2016-10-28</td>
      <td>1333.000000</td>
      <td>44.125000</td>
      <td>44.250000</td>
      <td>5.000000</td>
      <td>4.375000</td>
      <td>1.500000</td>
      <td>0.001250</td>
    </tr>
    <tr>
      <th>260</th>
      <td>2016-10-29</td>
      <td>883.777778</td>
      <td>41.333333</td>
      <td>42.666667</td>
      <td>4.600000</td>
      <td>9.222222</td>
      <td>1.444444</td>
      <td>0.013333</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2016-10-30</td>
      <td>859.285714</td>
      <td>45.714286</td>
      <td>43.428571</td>
      <td>3.714286</td>
      <td>3.714286</td>
      <td>1.000000</td>
      <td>-0.022857</td>
    </tr>
    <tr>
      <th>262</th>
      <td>2016-10-31</td>
      <td>30546.600000</td>
      <td>45.200000</td>
      <td>48.600000</td>
      <td>3.333333</td>
      <td>3.333333</td>
      <td>1.400000</td>
      <td>0.034000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dates of the debates, getting the index

sep26_df = poll_start_df[poll_start_df['Start Date'] == "2016-09-26"].index.values
oct9_df = poll_start_df[poll_start_df['Start Date'] == "2016-10-09"].index.values
oct19_df = poll_start_df[poll_start_df['Start Date'] == "2016-10-19"].index.values
```


```python
# Start with original figure
fig = poll_start_df.plot('Start Date','Difference',figsize=(12,4),
                         marker='o',linestyle='-',color='purple',xlim=(225,265))


# Now add the debate markers
plt.axvline(x=sep26_df, linewidth=4, color='grey')
plt.axvline(x=oct9_df, linewidth=4, color='grey')
plt.axvline(x=oct19_df, linewidth=4, color='grey')
```




    <matplotlib.lines.Line2D at 0x13b7cf6d8>




![png](output_31_1.png)


**RESULTS**

- After the first debate, Trump continued to do better.
- After the second debate, nobody got an immediate change in their position.
- After the last debate, Clinton did better.

---

## Using the New York Times API

I created a script that takes the date and topic and output all the articles on the date

- Will find relevant articles that happened after the debate to check if we can find articles that have relevancy to the NY Times.


```python
from nytimes import NYTimes
```


```python
# Make you sure you include your desired dates
# Politics is the topics that it will look for

data = NYTimes(9,27,2016,"Politics")
```

    These are the relevant topics: 
    Headline: Trump on the Fed’s Motives  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004674092/trump-on-the-feds-motives.html 
    
    Headline: Trump and Clinton Discuss Alicia Machado  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004674290/trump-and-clinton-discuss-alicia-machado.html 
    
    Headline: Trump and Clinton Clash Over Trade  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004674128/trump-and-clinton-clash-over-trade.html 
    
    Headline: Fact-Checking the First Presidential Debate  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004673274/fact-checking-the-first-presidential-debate.html 
    
    Headline: Holt’s Moderation Gets Mixed Reviews  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004673207/holts-moderation-gets-mixed-reviews.html 
    
    Headline: Clinton Wary of Trump and Nuclear Arms  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004673213/clinton-wary-of-trump-and-nuclear-arms.html 
    
    Headline: Trump Talks Taxes and Clinton’s Emails  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004673201/trump-talks-taxes-and-clintons-emails.html 
    
    Headline: Transcript of the First Debate  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/transcript-debate.html 
    
    Headline: Rudolph Giuliani Says Donald Trump Bit His Tongue   
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/rudy-giuliani-chelsea-clinton.html 
    
    Headline: How Hillary Clinton Went From Hesitant to Scorchin  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/hillary-clinton-donald-trump.html 
    
    Headline: Clinton Could Have Corrected Trump, but He Blared   
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/assertions-hofstra-debate.html 
    
    Headline: Lester Holt, Given a Choice Assignment, Opted for   
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/lester-holt-moderator.html 
    
    Headline: Hillary Clinton and Donald Trump Press Pointed Att  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/presidential-debate.html 
    
    Headline: Trump and Clinton in First Debate  
     Website can be found: https://www.nytimes.com/video/us/politics/100000004672976/trump-and-clinton-face-off-in-first-debate.html 
    
    Headline: Our Fact Checks of the First Debate  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/fact-check-debate.html 
    
    Headline: How The New York Times Will Fact-Check the Debate  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/new-york-times-debate-coverage.html 
    
    Headline: For Debate Organizer, a New Priority: Crowd Contro  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/crowd-campaign-debate.html 
    
    Headline: The What-Ifs: 11 Debate Possibilities That Should   
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/what-if-debate.html 
    
    Headline: Besieged Globalists Ponder What Went Wrong  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/globalism-un-assembly-nationalism-populism.html 
    
    Headline: Congress Meets, With 5 Days to Avoid Shutdown Over  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/congress-shutdown.html 
    
    Headline: How to Watch the First Debate and How It Works  
     Website can be found: https://www.nytimes.com/2016/09/27/us/politics/how-to-watch-debate-schedule.html 
    
    Headline: Did You Miss the Presidential Debate? Here Are the  
     Website can be found: https://www.nytimes.com/2016/09/26/us/politics/presidential-debate.html 
    
    


---

# Donor Data Set

The questions we will be trying to answer while looking at this Data Set is:

1. How much was donated and what was the average donation?
2. How did the donations differ between candidates?
3. How did the donations differ between Democrats and Republicans?
4. What were the demographics of the donors?
5. Is there a pattern to donation amounts?


```python
# Set the DataFrame as the csv file of NEW YORK
donor_df = pd.read_csv('Election_Donar_NY.csv', low_memory=False, encoding='utf-8', index_col=None)
```


```python
donor_df.shape
```




    (649460, 18)




```python
donor_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 649460 entries, C00575795 to C00575795
    Data columns (total 18 columns):
    cmte_id              649460 non-null object
    cand_id              649460 non-null object
    cand_nm              649460 non-null object
    contbr_nm            649456 non-null object
    contbr_city          649460 non-null object
    contbr_st            649377 non-null object
    contbr_zip           560658 non-null object
    contbr_employer      642406 non-null object
    contbr_occupation    649460 non-null float64
    contb_receipt_amt    649460 non-null object
    contb_receipt_dt     8149 non-null object
    receipt_desc         108347 non-null object
    memo_cd              251346 non-null object
    memo_text            649460 non-null object
    form_tp              649460 non-null int64
    file_num             649460 non-null object
    tran_id              648770 non-null object
    election_tp          0 non-null float64
    dtypes: float64(2), int64(1), object(15)
    memory usage: 94.1+ MB



```python
donor_df.describe()
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
      <th>contbr_occupation</th>
      <th>form_tp</th>
      <th>election_tp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.494600e+05</td>
      <td>6.494600e+05</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.643009e+02</td>
      <td>1.105477e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.576947e+04</td>
      <td>2.806734e+04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.010000e+04</td>
      <td>1.003942e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.500000e+01</td>
      <td>1.079445e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.700000e+01</td>
      <td>1.104813e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000e+02</td>
      <td>1.133832e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.277771e+07</td>
      <td>1.146285e+06</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
donor_df.columns
```




    Index(['cmte_id', 'cand_id', 'cand_nm', 'contbr_nm', 'contbr_city',
           'contbr_st', 'contbr_zip', 'contbr_employer', 'contbr_occupation',
           'contb_receipt_amt', 'contb_receipt_dt', 'receipt_desc', 'memo_cd',
           'memo_text', 'form_tp', 'file_num', 'tran_id', 'election_tp'],
          dtype='object')




```python
donor_df.reset_index(inplace=True)
donor_df.columns = ['cmte_id', 'cand_id', 'cand_nm', 'contbr_nm', 'contbr_city',
       'contbr_st', 'contbr_zip', 'contbr_employer', 'contbr_occupation',
       'contb_receipt_amt', 'contb_receipt_dt', 'receipt_desc', 'memo_cd',
       'memo_text', 'form_tp', 'file_num', 'tran_id', 'election_tp', 'delete']
```


```python
del donor_df['delete']
```


```python
donor_df.head()
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
      <th>cmte_id</th>
      <th>cand_id</th>
      <th>cand_nm</th>
      <th>contbr_nm</th>
      <th>contbr_city</th>
      <th>contbr_st</th>
      <th>contbr_zip</th>
      <th>contbr_employer</th>
      <th>contbr_occupation</th>
      <th>contb_receipt_amt</th>
      <th>contb_receipt_dt</th>
      <th>receipt_desc</th>
      <th>memo_cd</th>
      <th>memo_text</th>
      <th>form_tp</th>
      <th>file_num</th>
      <th>tran_id</th>
      <th>election_tp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>JONES TAKATA, LOUISE</td>
      <td>NEW YORK</td>
      <td>NY</td>
      <td>100162783</td>
      <td>NaN</td>
      <td>RETIRED</td>
      <td>100.00</td>
      <td>15-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4732422</td>
      <td>P2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>CODY, ERIN</td>
      <td>BUFFALO</td>
      <td>NY</td>
      <td>142221910</td>
      <td>RUPP BAASE PFALZGRAF CUNNINGHAM LLC</td>
      <td>ATTORNEY</td>
      <td>66.95</td>
      <td>24-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4752463</td>
      <td>P2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00577130</td>
      <td>P60007168</td>
      <td>Sanders, Bernard</td>
      <td>KEITH, SUSAN H</td>
      <td>NEW YORK</td>
      <td>NY</td>
      <td>100133107</td>
      <td>NOT EMPLOYED</td>
      <td>NOT EMPLOYED</td>
      <td>50.00</td>
      <td>06-MAR-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
      <td>SA17A</td>
      <td>1077404</td>
      <td>VPF7BKZ1KR1</td>
      <td>P2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00577130</td>
      <td>P60007168</td>
      <td>Sanders, Bernard</td>
      <td>LEPAGE, WILLIAM</td>
      <td>BROOKLYN</td>
      <td>NY</td>
      <td>112381202</td>
      <td>NEW YORK UNIVERSITY</td>
      <td>UNDERGRADUATE ADMINISTRATOR</td>
      <td>15.00</td>
      <td>04-MAR-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
      <td>SA17A</td>
      <td>1077404</td>
      <td>VPF7BKWHRY0</td>
      <td>P2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>BIELAT, VEDORA</td>
      <td>PLATTSBURGH</td>
      <td>NY</td>
      <td>129011729</td>
      <td>INFORMATION REQUESTED</td>
      <td>INFORMATION REQUESTED</td>
      <td>100.00</td>
      <td>12-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4714688</td>
      <td>P2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
# per amount of money
donor_df['contb_receipt_amt'].value_counts().head(10)
```




    25.0      94311
    50.0      72644
    100.0     67365
    10.0      55260
    5.0       42681
    15.0      28956
    27.0      28112
    250.0     26212
    2700.0    12947
    19.0      12348
    Name: contb_receipt_amt, dtype: int64




```python
don_rec_avg = donor_df['contb_receipt_amt'].mean()
don_rec_std = donor_df['contb_receipt_amt'].std()

print("The average is {0:.2f} and the standard deviation is {1:.2f}".format(don_rec_avg, don_rec_std))
```

    The average is 264.30 and the standard deviation is 25769.47


- Notice how the standard deviation is VERY large compared to the mean


```python
top_donor_df = donor_df['contb_receipt_amt'].copy()
top_donor_df.sort_values().head(10)
```




    177328   -10100.0
    408863    -9300.0
    409141    -7300.0
    6811      -6700.0
    646922    -6579.0
    408862    -6400.0
    4559      -5400.0
    353058    -5400.0
    457735    -5400.0
    4791      -5400.0
    Name: contb_receipt_amt, dtype: float64




```python
# NOTICE: There are huge negative values bc these are refunds
# We do not need this information
# this is why we have large stanard deviation
```


```python
top_donor_df = top_donor_df[top_donor_df > 0] # only for positive values
top_donor_df.sort_values().head()
```




    426236    0.01
    405767    0.04
    397171    0.04
    151779    0.08
    435352    0.09
    Name: contb_receipt_amt, dtype: float64




```python
# The top donation amount
top_donor_df.value_counts().head(10)
```




    25.0      94311
    50.0      72644
    100.0     67365
    10.0      55260
    5.0       42681
    15.0      28956
    27.0      28112
    250.0     26212
    2700.0    12947
    19.0      12348
    Name: contb_receipt_amt, dtype: int64




```python
common_don = top_donor_df[top_donor_df < 2500] # within the largest donation, 2500 was the largest

common_don.hist(bins=100) # There are peaks that our shown with value_counts() as well
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18b11f208>




![png](output_54_1.png)



```python
# Candidates of the election
candidate = donor_df['cand_nm'].unique()
candidate
```




    array(['Clinton, Hillary Rodham', 'Sanders, Bernard', 'Trump, Donald J.',
           "O'Malley, Martin Joseph", "Cruz, Rafael Edward 'Ted'",
           'Walker, Scott', 'Bush, Jeb', 'Rubio, Marco', 'Kasich, John R.',
           'Christie, Christopher J.', 'Stein, Jill', 'Johnson, Gary',
           'Graham, Lindsey O.', 'Webb, James Henry Jr.',
           'Carson, Benjamin S.', 'Paul, Rand', 'Fiorina, Carly',
           'Santorum, Richard J.', 'Jindal, Bobby', 'Huckabee, Mike',
           'Pataki, George E.', 'Gilmore, James S III', 'Lessig, Lawrence',
           'Perry, James R. (Rick)', 'McMullin, Evan'], dtype=object)




```python
# Dictionary of party affiliation
party_map = {'Clinton, Hillary Rodham': 'Democrat',
            'Sanders, Bernard': 'Democrat',
            'Trump, Donald J.': 'Republican',
            "O'Malley, Martin Joseph": 'Democrat',
            "Cruz, Rafael Edward 'Ted'": 'Republican',
            'Walker, Scott': 'Republican',
            'Bush, Jeb': 'Republican',
            'Rubio, Marco': 'Republican',
            'Christie, Christopher J.': 'Republican',
            'Stein, Jill': 'Green',
            'Johnson, Gary': 'Libertarian',
            'Graham, Lindsey O.': 'Republican',
            'Webb, James Henry Jr': 'Democrat',
            'Carson, Benjamin S.': 'Republican',
            'Paul, Rand': 'Republican',
            'Fiorina, Carly': 'Republican',
            'Santorum, Richard J.': 'Republican',
            'Jindal, Bobby': 'Republican',
            'Huckabee, Mike': 'Republican',
            'Pataki, George E.': 'Republican',
            'Gilmore, James S III': 'Republican',
            'Lessig, Lawrence': 'Democrat',
            'Perry, James R. (Rick)': 'Republican',
            'McMullin, Evan': 'Indepedent'
            }

# Now map the party with candidate
donor_df['Party'] = donor_df.cand_nm.map(party_map)

```


```python
# Redo the procedure but updated with the party affiliation
donor_df = donor_df[donor_df.contb_receipt_amt > 0]

```


```python
donor_df.head()
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
      <th>cmte_id</th>
      <th>cand_id</th>
      <th>cand_nm</th>
      <th>contbr_nm</th>
      <th>contbr_city</th>
      <th>contbr_st</th>
      <th>contbr_zip</th>
      <th>contbr_employer</th>
      <th>contbr_occupation</th>
      <th>contb_receipt_amt</th>
      <th>contb_receipt_dt</th>
      <th>receipt_desc</th>
      <th>memo_cd</th>
      <th>memo_text</th>
      <th>form_tp</th>
      <th>file_num</th>
      <th>tran_id</th>
      <th>election_tp</th>
      <th>Party</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>JONES TAKATA, LOUISE</td>
      <td>NEW YORK</td>
      <td>NY</td>
      <td>100162783</td>
      <td>NaN</td>
      <td>RETIRED</td>
      <td>100.00</td>
      <td>15-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4732422</td>
      <td>P2016</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>CODY, ERIN</td>
      <td>BUFFALO</td>
      <td>NY</td>
      <td>142221910</td>
      <td>RUPP BAASE PFALZGRAF CUNNINGHAM LLC</td>
      <td>ATTORNEY</td>
      <td>66.95</td>
      <td>24-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4752463</td>
      <td>P2016</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00577130</td>
      <td>P60007168</td>
      <td>Sanders, Bernard</td>
      <td>KEITH, SUSAN H</td>
      <td>NEW YORK</td>
      <td>NY</td>
      <td>100133107</td>
      <td>NOT EMPLOYED</td>
      <td>NOT EMPLOYED</td>
      <td>50.00</td>
      <td>06-MAR-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
      <td>SA17A</td>
      <td>1077404</td>
      <td>VPF7BKZ1KR1</td>
      <td>P2016</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00577130</td>
      <td>P60007168</td>
      <td>Sanders, Bernard</td>
      <td>LEPAGE, WILLIAM</td>
      <td>BROOKLYN</td>
      <td>NY</td>
      <td>112381202</td>
      <td>NEW YORK UNIVERSITY</td>
      <td>UNDERGRADUATE ADMINISTRATOR</td>
      <td>15.00</td>
      <td>04-MAR-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
      <td>SA17A</td>
      <td>1077404</td>
      <td>VPF7BKWHRY0</td>
      <td>P2016</td>
      <td>Democrat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00575795</td>
      <td>P00003392</td>
      <td>Clinton, Hillary Rodham</td>
      <td>BIELAT, VEDORA</td>
      <td>PLATTSBURGH</td>
      <td>NY</td>
      <td>129011729</td>
      <td>INFORMATION REQUESTED</td>
      <td>INFORMATION REQUESTED</td>
      <td>100.00</td>
      <td>12-APR-16</td>
      <td>NaN</td>
      <td>X</td>
      <td>* HILLARY VICTORY FUND</td>
      <td>SA18</td>
      <td>1091718</td>
      <td>C4714688</td>
      <td>P2016</td>
      <td>Democrat</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Represents how many people dondated for them

donor_df.groupby('cand_nm')['contb_receipt_amt'].count()
```




    cand_nm
    Bush, Jeb                      2321
    Carson, Benjamin S.            6566
    Christie, Christopher J.        466
    Clinton, Hillary Rodham      394676
    Cruz, Rafael Edward 'Ted'     16147
    Fiorina, Carly                 1206
    Gilmore, James S III              5
    Graham, Lindsey O.              293
    Huckabee, Mike                  240
    Jindal, Bobby                    21
    Johnson, Gary                   781
    Kasich, John R.                1330
    Lessig, Lawrence                116
    McMullin, Evan                  103
    O'Malley, Martin Joseph         338
    Pataki, George E.               181
    Paul, Rand                     1130
    Perry, James R. (Rick)           27
    Rubio, Marco                   4487
    Sanders, Bernard             173387
    Santorum, Richard J.             69
    Stein, Jill                    1001
    Trump, Donald J.              35762
    Walker, Scott                   231
    Webb, James Henry Jr.            46
    Name: contb_receipt_amt, dtype: int64




```python
# Represents total amount of donation

total_cand_sum = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()
```


```python
total_cand_sum.plot(kind='bar')

# Obama got the most donation because he is the only in the democrats pary
```




    <matplotlib.axes._subplots.AxesSubplot at 0x158d24f28>




![png](output_61_1.png)



```python
# donation of democrats vs republicans
donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x158240518>




![png](output_62_1.png)



```python
occupations_df = donor_df.pivot_table('contb_receipt_amt', 
                                     index='contbr_occupation',
                                     columns='Party',
                                     aggfunc='sum')
occupations_df.head()
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
      <th>Party</th>
      <th>Democrat</th>
      <th>Green</th>
      <th>Indepedent</th>
      <th>Libertarian</th>
      <th>Republican</th>
    </tr>
    <tr>
      <th>contbr_occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADMINISTRATIVE ASSISTANT</th>
      <td>150.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ATTORNEY</th>
      <td>290.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CHARITY CONSULTANT</th>
      <td>250.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EDUCATOR</th>
      <td>67.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HEALTHCARE MANAGER</th>
      <td>34.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
occupations_df.shape

# LARGE DF because there are a lot of diff. contribution
```




    (17095, 5)




```python
# The 1 represents bc we are adding the column
# and checking if the total sum of that occupation is greater than 1000000
# And only working with those occupation
occupations_df = occupations_df[occupations_df.sum(1) > 1000000]
```


```python
occupations_df.shape
```




    (15, 5)




```python
occupations_df.plot(kind='bar')
plt.show()
```


![png](output_67_0.png)



```python
# horizontal graph and seismic plots the blue and red
occupations_df.plot(kind='barh',figsize=(10,12), cmap='seismic')
plt.show()

# FIX the Information columns as they are like an NA

```


![png](output_68_0.png)



```python
# removing the info. columns
occupations_df.drop(['INFORMATION REQUESTED'], axis=0, inplace=True) # columns
```


```python
# horizontal graph
occupations_df.plot(kind='barh',figsize=(10,12), cmap='seismic')
plt.show()
```


![png](output_70_0.png)


For more on general data analysis of politics, I highly suggest the https://fivethirtyeight.com/politics/ website!


