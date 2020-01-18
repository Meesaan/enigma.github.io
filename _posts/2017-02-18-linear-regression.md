---
title: "Linear Regression"
category: ML
tags: [Data Science]
date: 2018-02-18
header:
  image: "/images/lr1.jpg"
excerpt: "Data Science, Supervised Learning"
---


## Intuitions Behind Linear Regression Models: Daimonds Dataset.

### 1. Introduction
### What is Linear Regression?  
Linear regression is one of the most widely known modeling techniques in machine learning. It is usually among the first few topics people pick while learning predictive modeling for supervised learning. In this type of regression, the dependent variables are continuous values while the independent variables can be continuous, discrete or even categorical. The nature of regression line is linear in shape. We can therfore define linear regression analysis as a method of ***predicting continous values***, after investigating a ***linear relationship*** between independent variables -***X*** and a response variable-***Y***.
> Mathematically, linear regression model is expressed as:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <img src="formula.png">  
where y<sub>i</sub> is an instance of y, α -the intercept of the line, indicating the value of Y when X is equal to zero, β -its slope, ε<sub>i</sub> -instance error and X-independent variable. α and β are generally referred to as the coefficients.

It does this by fitting a straight line to the data points in such a way that the differences between the distances of the data points from the line is minimised. Let me explain what this means literarily. Say we plot a graph between carats and price from the diamonds dataset. We may get one as shown below.
<img src="linear_regg.png">
The linear line, known as best fit line, tries to draw a line in such a way that it passes through all the data points. In this case, it cannot do so without the lines becoming curvy or non-linear (This would be a non-linear regression which is out of scope of this study). It finds the best straight line that is as closest to the points as possible. The difference between the best fit line and a datapoint is known as the error or residual. The line created above is therefore the  best line it could get with the least amount of error - hence the name "best fit line". This right here is what linear regression is all about- minimising the total amount of error between datapoints and its best fit line. Your next question should be: How do we find the best values of alpha and beta in such a way that the error is minimised in order to achieve the best fit line?  
In this tutorial, we will use the diamonds dataset to answer this question for predicting the price of diamonds based on a number of predictors. The main objective of this project is to review three major methods used for finding best fit line for a linear regression model in the most simplest of ways, such that a new data science enthusiast can understand easily. The three methods we would be discussing are: Ordinary Least Squares, Gradient Descent and Regularization. But first, we have to import, clean the data and prepare it for modelling before getting to the interesting stuff.
You can find more description of the dataset in [this link](https://www.kaggle.com/shivam2503/diamonds)

### 2.  Some Data Analysis


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25744</td>
      <td>1.51</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS1</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>14674</td>
      <td>7.35</td>
      <td>7.29</td>
      <td>4.55</td>
    </tr>
    <tr>
      <td>51528</td>
      <td>0.83</td>
      <td>Ideal</td>
      <td>F</td>
      <td>SI2</td>
      <td>59.5</td>
      <td>57.0</td>
      <td>2386</td>
      <td>6.13</td>
      <td>6.10</td>
      <td>3.64</td>
    </tr>
    <tr>
      <td>42030</td>
      <td>0.46</td>
      <td>Good</td>
      <td>E</td>
      <td>VS2</td>
      <td>60.9</td>
      <td>62.0</td>
      <td>1267</td>
      <td>4.93</td>
      <td>4.96</td>
      <td>3.01</td>
    </tr>
    <tr>
      <td>50796</td>
      <td>0.71</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS1</td>
      <td>61.8</td>
      <td>58.0</td>
      <td>2306</td>
      <td>5.72</td>
      <td>5.76</td>
      <td>3.55</td>
    </tr>
    <tr>
      <td>42914</td>
      <td>0.51</td>
      <td>Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>58.7</td>
      <td>58.0</td>
      <td>1363</td>
      <td>5.21</td>
      <td>5.25</td>
      <td>3.07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 10 columns):
    carat      53940 non-null float64
    cut        53940 non-null object
    color      53940 non-null object
    clarity    53940 non-null object
    depth      53940 non-null float64
    table      53940 non-null float64
    price      53940 non-null int64
    x          53940 non-null float64
    y          53940 non-null float64
    z          53940 non-null float64
    dtypes: float64(6), int64(1), object(3)
    memory usage: 4.1+ MB


Things to note; We have aproximately 54,000 instances of diamonds, no missing values and there are three object values we need to change to numeric.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.797940</td>
      <td>61.749405</td>
      <td>57.457184</td>
      <td>3932.799722</td>
      <td>5.731157</td>
      <td>5.734526</td>
      <td>3.538734</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.474011</td>
      <td>1.432621</td>
      <td>2.234491</td>
      <td>3989.439738</td>
      <td>1.121761</td>
      <td>1.142135</td>
      <td>0.705699</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.200000</td>
      <td>43.000000</td>
      <td>43.000000</td>
      <td>326.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.400000</td>
      <td>61.000000</td>
      <td>56.000000</td>
      <td>950.000000</td>
      <td>4.710000</td>
      <td>4.720000</td>
      <td>2.910000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.700000</td>
      <td>61.800000</td>
      <td>57.000000</td>
      <td>2401.000000</td>
      <td>5.700000</td>
      <td>5.710000</td>
      <td>3.530000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.040000</td>
      <td>62.500000</td>
      <td>59.000000</td>
      <td>5324.250000</td>
      <td>6.540000</td>
      <td>6.540000</td>
      <td>4.040000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5.010000</td>
      <td>79.000000</td>
      <td>95.000000</td>
      <td>18823.000000</td>
      <td>10.740000</td>
      <td>58.900000</td>
      <td>31.800000</td>
    </tr>
  </tbody>
</table>
</div>



.describe method provides some measures of variability (spread) from the numerical variables. We can note that;
1. Max number of carat in this dataset is 5 (ughh... where are the 33 carats of Elizabeth Taylor's??? or the 18 carats of Beyonce? can you feel my disappointment? Anyway, at the end of this study, you can get their independent variables online and predict the prices. See how close to the actual price you get (this is EXCERSISE ONE.)).  
2. A large majority of the price of diamonds, 75%, are sold at 5,000USD or less.
3. The average depth of diamond is 62 with its values very close to the mean.
4. this is not the case for price that is relatively far spread out from the mean.
5. Notice the min values for X, Y, Z? That tells me there are missing values.


```python
correlation = df.corr()

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(correlation, center=0,  square=True,
                annot=True)
plt.show()
```


![png](output_10_0.png)





```python
g = sns.pairplot(df)
```


![png](output_12_0.png)


1. If you take a look at price, the bar chart is skewed to the right,signifying that most of the values counts at 500USD. Variables Y and Z are positively strongly correlated with price, meaning that as witdth and depth increases, price also increases. Carat is ever so slightly correlated with price while depth as well as table have little or no correlation with price.
3. Carat weight distribution is also right skewd portraying that majority of diamonds in the population is on avearge low carat weight.
2. Some independentvariables are correlated with each other (multicollinearity) which affects prediction. They are X,Y,Z, carat,X,Y,Z and depth,table(showing high negative correlation).


```python
sns.catplot(x='cut', data=df , kind='count',aspect=2.5 )
```




    <seaborn.axisgrid.FacetGrid at 0x1986331d248>




![png](output_14_1.png)



```python
sns.catplot(x='color', data=df , kind='count',aspect=2.5 )
```




    <seaborn.axisgrid.FacetGrid at 0x19863c12ac8>




![png](output_15_1.png)



```python
sns.catplot(x='clarity', data=df , kind='count',aspect=2.5 )
```




    <seaborn.axisgrid.FacetGrid at 0x1986889cb48>




![png](output_16_1.png)



```python
sns.catplot(x='cut', y='price', data=df, kind='box' ,aspect=2.5 )
```




    <seaborn.axisgrid.FacetGrid at 0x19868883d08>




![png](output_17_1.png)



```python
sns.catplot(x='color', y='price' , data=df , kind='box', aspect=2.5)
```




    <seaborn.axisgrid.FacetGrid at 0x1986501eac8>




![png](output_18_1.png)



```python
sns.catplot(x='clarity', y='price', data=df, kind='box' ,aspect=2.5 )
```




    <seaborn.axisgrid.FacetGrid at 0x198650add08>




![png](output_19_1.png)


### 3. Data Cleaning and Preparation


```python
df.head()
df.shape
```




    (53940, 10)



Data preparation steps:
1. Drop price.
2. Transform price to log(price) to reduce its skewness.
3. Split data into train and test sets.
4. Create pipeline
    - Find instances with missing values
    - Convert cut, color and clarity to numeric values


```python
X = df.drop(['price', 'y', 'z'], axis=1).copy()
```


```python
y = df['price'].copy()
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
X_train.shape
```




    (37758, 7)




```python
df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2207</td>
      <td>1.00</td>
      <td>Premium</td>
      <td>G</td>
      <td>SI2</td>
      <td>59.1</td>
      <td>59.0</td>
      <td>3142</td>
      <td>6.55</td>
      <td>6.48</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2314</td>
      <td>1.01</td>
      <td>Premium</td>
      <td>H</td>
      <td>I1</td>
      <td>58.1</td>
      <td>59.0</td>
      <td>3167</td>
      <td>6.66</td>
      <td>6.60</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4791</td>
      <td>1.10</td>
      <td>Premium</td>
      <td>G</td>
      <td>SI2</td>
      <td>63.0</td>
      <td>59.0</td>
      <td>3696</td>
      <td>6.50</td>
      <td>6.47</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5471</td>
      <td>1.01</td>
      <td>Premium</td>
      <td>F</td>
      <td>SI2</td>
      <td>59.2</td>
      <td>58.0</td>
      <td>3837</td>
      <td>6.50</td>
      <td>6.47</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>10167</td>
      <td>1.50</td>
      <td>Good</td>
      <td>G</td>
      <td>I1</td>
      <td>64.0</td>
      <td>61.0</td>
      <td>4731</td>
      <td>7.15</td>
      <td>7.04</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>11182</td>
      <td>1.07</td>
      <td>Ideal</td>
      <td>F</td>
      <td>SI2</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4954</td>
      <td>0.00</td>
      <td>6.62</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>11963</td>
      <td>1.00</td>
      <td>Very Good</td>
      <td>H</td>
      <td>VS2</td>
      <td>63.3</td>
      <td>53.0</td>
      <td>5139</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>13601</td>
      <td>1.15</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS2</td>
      <td>59.2</td>
      <td>56.0</td>
      <td>5564</td>
      <td>6.88</td>
      <td>6.83</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>15951</td>
      <td>1.14</td>
      <td>Fair</td>
      <td>G</td>
      <td>VS1</td>
      <td>57.5</td>
      <td>67.0</td>
      <td>6381</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>24394</td>
      <td>2.18</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>59.4</td>
      <td>61.0</td>
      <td>12631</td>
      <td>8.49</td>
      <td>8.45</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>24520</td>
      <td>1.56</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS2</td>
      <td>62.2</td>
      <td>54.0</td>
      <td>12800</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>26123</td>
      <td>2.25</td>
      <td>Premium</td>
      <td>I</td>
      <td>SI1</td>
      <td>61.3</td>
      <td>58.0</td>
      <td>15397</td>
      <td>8.52</td>
      <td>8.42</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>26243</td>
      <td>1.20</td>
      <td>Premium</td>
      <td>D</td>
      <td>VVS1</td>
      <td>62.1</td>
      <td>59.0</td>
      <td>15686</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>27112</td>
      <td>2.20</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI1</td>
      <td>61.2</td>
      <td>59.0</td>
      <td>17265</td>
      <td>8.42</td>
      <td>8.37</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>27429</td>
      <td>2.25</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>62.8</td>
      <td>59.0</td>
      <td>18034</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>27503</td>
      <td>2.02</td>
      <td>Premium</td>
      <td>H</td>
      <td>VS2</td>
      <td>62.7</td>
      <td>53.0</td>
      <td>18207</td>
      <td>8.02</td>
      <td>7.95</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>27739</td>
      <td>2.80</td>
      <td>Good</td>
      <td>G</td>
      <td>SI2</td>
      <td>63.8</td>
      <td>58.0</td>
      <td>18788</td>
      <td>8.90</td>
      <td>8.85</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>49556</td>
      <td>0.71</td>
      <td>Good</td>
      <td>F</td>
      <td>SI2</td>
      <td>64.1</td>
      <td>60.0</td>
      <td>2130</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>49557</td>
      <td>0.71</td>
      <td>Good</td>
      <td>F</td>
      <td>SI2</td>
      <td>64.1</td>
      <td>60.0</td>
      <td>2130</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>51506</td>
      <td>1.12</td>
      <td>Premium</td>
      <td>G</td>
      <td>I1</td>
      <td>60.4</td>
      <td>59.0</td>
      <td>2383</td>
      <td>6.71</td>
      <td>6.67</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing = df.loc[(df['x']==0) & (df['y']==0) & (df['z']==0)].index
df.drop(missing, inplace=True)
```


```python
print(len(df[(df['x']==0) | (df['y']==0) | (df['z']==0)])) #we can fill these with the mean
```

    13



```python
miss = df[(df['x']==0) | (df['y']==0) | (df['z']==0)]
miss.replace(0, np.NaN)# I am replacing 0 to NaN, just so the imputer finds it.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2207</td>
      <td>1.00</td>
      <td>Premium</td>
      <td>G</td>
      <td>SI2</td>
      <td>59.1</td>
      <td>59.0</td>
      <td>3142</td>
      <td>6.55</td>
      <td>6.48</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2314</td>
      <td>1.01</td>
      <td>Premium</td>
      <td>H</td>
      <td>I1</td>
      <td>58.1</td>
      <td>59.0</td>
      <td>3167</td>
      <td>6.66</td>
      <td>6.60</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4791</td>
      <td>1.10</td>
      <td>Premium</td>
      <td>G</td>
      <td>SI2</td>
      <td>63.0</td>
      <td>59.0</td>
      <td>3696</td>
      <td>6.50</td>
      <td>6.47</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5471</td>
      <td>1.01</td>
      <td>Premium</td>
      <td>F</td>
      <td>SI2</td>
      <td>59.2</td>
      <td>58.0</td>
      <td>3837</td>
      <td>6.50</td>
      <td>6.47</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>10167</td>
      <td>1.50</td>
      <td>Good</td>
      <td>G</td>
      <td>I1</td>
      <td>64.0</td>
      <td>61.0</td>
      <td>4731</td>
      <td>7.15</td>
      <td>7.04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>11182</td>
      <td>1.07</td>
      <td>Ideal</td>
      <td>F</td>
      <td>SI2</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4954</td>
      <td>NaN</td>
      <td>6.62</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>13601</td>
      <td>1.15</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS2</td>
      <td>59.2</td>
      <td>56.0</td>
      <td>5564</td>
      <td>6.88</td>
      <td>6.83</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>24394</td>
      <td>2.18</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>59.4</td>
      <td>61.0</td>
      <td>12631</td>
      <td>8.49</td>
      <td>8.45</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>26123</td>
      <td>2.25</td>
      <td>Premium</td>
      <td>I</td>
      <td>SI1</td>
      <td>61.3</td>
      <td>58.0</td>
      <td>15397</td>
      <td>8.52</td>
      <td>8.42</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>27112</td>
      <td>2.20</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI1</td>
      <td>61.2</td>
      <td>59.0</td>
      <td>17265</td>
      <td>8.42</td>
      <td>8.37</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>27503</td>
      <td>2.02</td>
      <td>Premium</td>
      <td>H</td>
      <td>VS2</td>
      <td>62.7</td>
      <td>53.0</td>
      <td>18207</td>
      <td>8.02</td>
      <td>7.95</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>27739</td>
      <td>2.80</td>
      <td>Good</td>
      <td>G</td>
      <td>SI2</td>
      <td>63.8</td>
      <td>58.0</td>
      <td>18788</td>
      <td>8.90</td>
      <td>8.85</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>51506</td>
      <td>1.12</td>
      <td>Premium</td>
      <td>G</td>
      <td>I1</td>
      <td>60.4</td>
      <td>59.0</td>
      <td>2383</td>
      <td>6.71</td>
      <td>6.67</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Remove Outliers


```python

```

#### 3a. Create Numeric Pipeline


```python
from sklearn.pipeline import Pipeline

#This imputer is still experimental, however, since we have few missing values

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
train_num = X_train.select_dtypes(include=[np.number])

num_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler())
])

train_num_trx = num_pipeline.fit_transform(train_num)
train_num_trx
```




    array([[ 0.86265905, -0.31143736, -0.20709927,  1.05558107],
           [-1.02988861,  0.17854897, -0.65621302, -1.20773446],
           [ 0.86265905,  0.45854116, -0.20709927,  0.90409932],
           ...,
           [-0.987832  , -1.01141784,  0.24201448, -1.10971685],
           [ 0.21078152,  0.73853335,  0.69112823,  0.35163648],
           [ 0.7154609 , -0.94141979,  0.24201448,  0.96647416]])




```python
import category_encoders as ce

train_cat = X_train.select_dtypes(exclude=[np.number])

cat_pipeline = Pipeline([
    ('encoder', ce.OneHotEncoder()),
    ('scaler', StandardScaler())
])
train_cat_trx = cat_pipeline.fit_transform(train_cat)
train_cat_trx.shape

```




    (37758, 20)




```python
from sklearn.compose import ColumnTransformer as colt

num_feats = list(train_num)
cat_feats = list(train_cat)

all_pipeline = colt([
    ('num', num_pipeline, num_feats),
    ('cat', cat_pipeline, cat_feats)
])
```


```python
X_train = all_pipeline.fit_transform(X_train)
X_test = all_pipeline.transform(X_test)
```

### 4. Model Training

### 4a. Ordinary Least Squares.

Remember that our goal in a multiple regression task is to minimise error term, otherwise known as optimisation. Ordinary least squares is an optimisation technique that aims to do just that. It does this by calculating the vertical distance from each data point to the fit line, squares it and sums all the squared errors together. Sklearn linear regression model is based on OLS. There are several measures of errors, however, we will use RMSE (Root Mean Squared Error) in this project.


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear = LinearRegression()
linear.fit(X_train, y_train)
ols_predict = linear.predict(X_test)
ols_predict
```




    array([  732.67229206,  3202.35979206,  1942.35979206, ...,
           10613.35979206,  4118.35979206,  2014.67229206])




```python
print("Top 3 Labels:", list(y_test.head(3)))
print("Last 3 Labels:", list(y_test.tail(3)))
```

    Top 3 Labels: [559, 2201, 1238]
    Last 3 Labels: [13157, 2960, 1323]



```python
ols_mse = mean_squared_error(ols_predict, y_test)
ols_rmse = np.sqrt(ols_mse)
ols_rmse
```




    1115.6635561618737




```python
linear.intercept_, linear.coef_
```




    (3951.6149181773458,
     array([ 5.33540551e+03, -9.34592220e+01, -5.86361472e+01, -1.13654862e+03,
            -3.70779320e+14, -3.30258251e+14, -3.16454436e+14, -2.16914288e+14,
            -1.28625715e+14,  1.01715465e+15,  1.08138098e+15,  1.07350989e+15,
             8.41710323e+14,  1.13809152e+15,  9.28428007e+14,  6.23687032e+14,
            -3.03546520e+15, -4.37862131e+15, -3.74628598e+15, -3.91642063e+15,
            -4.48751669e+15, -2.63111932e+15, -1.86762595e+15, -1.20939832e+15]))



### 4b. Gradient Descent

Another method capable of finding optimal α and β is the gradient descent algorithm. The idea is to iteratively adjust the coefficients (or weights, or parameters of the line) with the aim of minimising the error (also called cost function). It starts by initialising weights the  with random values, then calculates the cost function, say with RMSE. Note how this is different from OLS where . Next, the weights are moved again ***ever so slightly*** from its initial random position where the cost function is calculated again. The difference between the initial error of step 1 and this step is known as the gradient. This is done over and over till the gradient is zero or its minimum, otherwise known as global minimum. (EXERCISE 2:WHAT DO YOU THINK LOCAL MINIMUM MEANS?). If we drew a graph of movements in cost fuction and weights (α, β = θ), it would look like this.
<img src ='cost.png'>
I said before that we take steps ever so slightly to move the weights. How does one determine the size of the steps then? First of all, size of steps taken is known as ***Learning Rate***. Secondly, you should note that if the learning rate is too small, then the algorithm will go through many iterations to reach convergence (global minimun) which s computationally expensive. Alternatively, if the learning rate is too large, you might miss the convergence point and find your local minimum at the other side climbing up.
There are two main types of gradient descent. Batch gradient descent and stochastic gradient descent...


```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=100, tol=-np.infty, penalty=None, eta0=0.002, random_state=42)
sgd.fit(X_train, y_train)
sgd_predict = sgd.predict(X_test)
sgd_predict
```




    array([  711.93272397,  3187.1186379 ,  1930.96304552, ...,
           10612.15254653,  4097.77917314,  1991.9725265 ])




```python
sgd_mse = mean_squared_error(sgd_predict, y_test)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse
```




    1115.3650978209096




```python
sgd.intercept_, sgd.coef_
```




    (array([3951.07972845]),
     array([ 5.32212749e+03, -8.91757741e+01, -6.01292412e+01, -1.11702070e+03,
             5.06678414e+01,  1.40677835e+01,  2.47294196e+00, -4.02907739e+01,
            -1.20314342e+02, -1.28945258e+02,  1.62386253e+02,  1.37863345e+02,
            -2.49853126e+02,  5.76616471e+01,  2.07749913e+02, -3.85839214e+02,
             2.70602364e+02,  1.01343287e+02,  2.03890578e+02, -4.94678669e+02,
            -1.48952740e+02,  2.57599601e+02,  2.45393928e+02, -4.62429991e+02]))



### 4c. Regularisation of Linear Regression.

A requirement for machine learning algorithm is to split a dataset into train and test sets as we had done earlier. We trained our model with the train set and made predictions with test sets that the model had not seen. Suppose we are comparing two machine learning models where one is OLS. We found that no matter how much we try to fit the line in training, it could never be curvy. In other words, it could never capture the true properties of the train dataset. This inability for OLS and other linear machine learning models is called ***Bias*** and make no mistake my friend, the bias is large. Suppose our other model during training is super flexible and captures the true nature of relationships in the dataset. If you calculate the RMSE, the second model would win because of its incredible low or zero bias. Next is to predict on test sets. When yu fit them to the test sets, OLS wins. This is because the second model overfit the train set as such could not generalise to a new (test) set. The difference in RMSE acquired during train and test sets is called ***Variance***. Our second hypothetical model has a high variance because it suggests that it is hard to predict how it would perform in generalisation while low variance produces consistent good predictions. Lets review these lines for better expalnation.
<img src = 'bias_var.png'>[Source](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)  

In machine learning, our goal is to find low bias and low variance (correct fit) models for our datasets. This is done by finding an optimal point between a simple model (OLS) and a complex model (non-linear). Some of the methods for finding this spot is Bagging, Boosting and Regularisation. In this section, we will treat Ridge, Lasso and ElasticNet.

### 4c.1. Ridge Regression
In the example above, we have assumed that the dataset presented to OLS has high bias. What about in situations where the line captures the true nature of the train dataset? In other words, the model has relatively low bias, high variance? Ridge regression aslo known as L2 regularisation, provides a solution. The idea of ridge regression is to find a line that does not fit to the train data so well by shrinking its parameters towards zero. Yes, it intentionally reduces fitness. This objective is to help it generalise well to unseen data. It does this by introducing a small amount of bias when fitting a line. Because of the introduced amount of bias, a significant drop in variance is also achieved. When OLS estimates the values of its parameters, it minimises error by summing the square residuals. However, in ridge regression, it does the same thing but adds  ***λ * β<sup>2</sup>***, also known as ridge regression penalty. <img src = 'ridge.png'>
By intuition, when λ is 0, we still have OLS. However, as λ increases, fitness reduces. How then can we decide the optimal value of λ? Different values are tried in cross-validation to determine the value that results in the least amount of variance.
We should not expect it to perform better because we suspect our dataset to have high bias already.


```python
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=[1.0, 100.0, 1000.0])#obviously 1.0 is the least alpha-regularisation term for the least variance. check with just the other values. See how rmse changes.
ridge.fit(X_train, y_train)
ridge_predict = ridge.predict(X_test)
ridge_predict
```




    array([  709.77330148,  3197.05202441,  1941.87437863, ...,
           10609.22298587,  4107.93222041,  1993.42786661])




```python
ridge_mse = mean_squared_error(ridge_predict, y_test)
ridge_rmse = np.sqrt(ridge_mse)
ridge_rmse
```




    1115.68103762031



### 4c.2. Lasso Regression
Lasso regression also known as L1 regularisation is somewhat similar to ridge regression in that they both add penalty to least squares. However, where ridge squares β, lasso estimates the absolute value of β. <img src = 'lasso.png'>
LASSO - Least Absolute Shrinkage and Selection Operator is a powerful method that perform two main tasks - regularisation and feature selection. The LASSO method puts a constraint on the sum of the absolute values of the model parameters. The sum has to be less than a fixed value (upper bound). In order to do so the method applies a shrinking (regularization) process where it penalizes the
coefficients of the regression variables shrinking some of them to zero. During
features selection process the variables that still have a non-zero coefficient
after the shrinking process are selected to be part of the model. The goal of
this process is to minimize the prediction error.
In practice the tuning parameter λ, that controls the strength of the penalty,
assume a great importance. Indeed when λ is sufficiently large then coefficients are forced to be exactly equal to zero, this way dimensionality can be
reduced. The larger is the parameter λ the more number of coefficients are
shrinked to zero. On the other hand if λ = 0 we have an OLS (Ordinary
Least Sqaure) regression.
lasso regression works best when your dataset contains useless variables. it helps to weed out the useless ones and keeps the important ones. since we do not have any useless variable, we can assume that our prediction would be similar to to ridge regression.


```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_pred
```




    array([  695.25222515,  3198.82712665,  1936.59810895, ...,
           10605.48267709,  4110.66266441,  1986.78899421])




```python
lasso_mse = mean_squared_error(lasso_pred, y_test)
lasso_rmse = np.sqrt(lasso_mse)
lasso_rmse
```




    1115.7219085433078



### 4c.3. Elastic-Net Regression
We concluded earlier that while ridge regression works best for dataset where most of the variables in the model are useful, lasso regression works best when most are useless. However, what do you do when your dataset has thousands of variables without knowing in advance the type of dataset? Elastic-net regression to the rescue. Just like ridge and Lasso, elastic regression starts with least squares, then combines the penalties of Ridge and Lasso together. However, Lasso and Ridge penalties get their own λs. Cross validation on different combinations of λs to find the best values.


```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(random_state=42)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)
elastic_pred
```




    array([ 302.5833142 , 3336.77284556, 1830.95342856, ..., 8828.92558911,
           4083.6517687 , 1871.85139248])




```python
elastic_mse = mean_squared_error(elastic_pred, y_test)
elastic_rmse = np.sqrt(elastic_mse)
elastic_rmse
```




    1616.2122400205462



### Conclusion.
Linear regression has some flaws, mainly because real life situations do not all posses linear relationshipit does not meet the low bias and variace criteria.



```python

```
