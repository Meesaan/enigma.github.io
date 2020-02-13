---
title: "Kaggle: Ames_Housing"
category: ML
tags: [Data Science]
date: 2018-02-18
header:
  image: "/images/ml1.jpg"
excerpt: "Data Science, Supervised Learning"
---

# This is a Kaggle project tutorial that predicts Ames Iowa houses.

Head to [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) to get a full description of this dataset.  


## 1. Generate an idea  
    Firstly, it is always good practice to have knowledge of what the objective of the project is. This means analysing who/what/where/when stands to benefit from your outcome. It helps you understand as well as create a guideline for the problem on a personal scale.  
    For this project, let us assume we are working for  real estate investors who would like to predict the SalePrice of houses in Ames, Iowa, given 80 factors or predictor variables. They would like to know if investing would be a good business idea. With this in mind, you can start thinking of types of information you could provide your employer that would help them gain competitive advantage. How would you deliver your findings to a non-science audience? You should develop these thought processes from the beginning.
    At this point, if you have previewed the data, you can tell that this is a batch univariate regression problem. Although RMSE has been chosen as its performance measure, note that if there are lots of outliers, you could also try MAE.  

## 2. Load the data


```python
#first, we load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
#load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sub_df = pd.read_csv("sample_submission.csv")

pd.set_option('display.max_columns', None)
train_df.sample(5)
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1038</th>
      <td>1039</td>
      <td>160</td>
      <td>RM</td>
      <td>21.0</td>
      <td>1533</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>MeadowV</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Twnhs</td>
      <td>2Story</td>
      <td>4</td>
      <td>6</td>
      <td>1970</td>
      <td>2008</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>546</td>
      <td>546</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>798</td>
      <td>546</td>
      <td>0</td>
      <td>1344</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>97000</td>
    </tr>
    <tr>
      <th>828</th>
      <td>829</td>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>28698</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Sev</td>
      <td>ClearCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>1967</td>
      <td>1967</td>
      <td>Flat</td>
      <td>Tar&amp;Grv</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>TA</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>LwQ</td>
      <td>249</td>
      <td>ALQ</td>
      <td>764</td>
      <td>0</td>
      <td>1013</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1160</td>
      <td>966</td>
      <td>0</td>
      <td>2126</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min2</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1967.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>538</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>486</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>225</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2009</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>185000</td>
    </tr>
    <tr>
      <th>174</th>
      <td>175</td>
      <td>20</td>
      <td>RL</td>
      <td>47.0</td>
      <td>12416</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1986</td>
      <td>1986</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>Plywood</td>
      <td>Stone</td>
      <td>132.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>Fa</td>
      <td>No</td>
      <td>ALQ</td>
      <td>1398</td>
      <td>LwQ</td>
      <td>208</td>
      <td>0</td>
      <td>1606</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1651</td>
      <td>0</td>
      <td>0</td>
      <td>1651</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min2</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1986.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>616</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>184000</td>
    </tr>
    <tr>
      <th>534</th>
      <td>535</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9056</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>Gd</td>
      <td>Av</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>707</td>
      <td>707</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>707</td>
      <td>707</td>
      <td>0</td>
      <td>1414</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2004.0</td>
      <td>Fin</td>
      <td>2</td>
      <td>403</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>100</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>178000</td>
    </tr>
    <tr>
      <th>443</th>
      <td>444</td>
      <td>120</td>
      <td>RL</td>
      <td>53.0</td>
      <td>3922</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Blmngtn</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>TwnhsE</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>WdShing</td>
      <td>Wd Shng</td>
      <td>BrkFace</td>
      <td>72.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>TA</td>
      <td>Av</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1258</td>
      <td>1258</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1258</td>
      <td>0</td>
      <td>0</td>
      <td>1258</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2007.0</td>
      <td>Fin</td>
      <td>3</td>
      <td>648</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>144</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>172500</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(train_df.shape)
print(test_df.shape)
```

    (1460, 81)
    (1459, 80)


## 3. Analyze and Visualize Data
This stage provides an oppurtunity to gain some meaningful insights and get a statistical 'feel' of hidden elements in the data.


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object
     6   Alley          91 non-null     object
     7   LotShape       1460 non-null   object
     8   LandContour    1460 non-null   object
     9   Utilities      1460 non-null   object
     10  LotConfig      1460 non-null   object
     11  LandSlope      1460 non-null   object
     12  Neighborhood   1460 non-null   object
     13  Condition1     1460 non-null   object
     14  Condition2     1460 non-null   object
     15  BldgType       1460 non-null   object
     16  HouseStyle     1460 non-null   object
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object
     22  RoofMatl       1460 non-null   object
     23  Exterior1st    1460 non-null   object
     24  Exterior2nd    1460 non-null   object
     25  MasVnrType     1452 non-null   object
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object
     28  ExterCond      1460 non-null   object
     29  Foundation     1460 non-null   object
     30  BsmtQual       1423 non-null   object
     31  BsmtCond       1423 non-null   object
     32  BsmtExposure   1422 non-null   object
     33  BsmtFinType1   1423 non-null   object
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object
     40  HeatingQC      1460 non-null   object
     41  CentralAir     1460 non-null   object
     42  Electrical     1459 non-null   object
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object
     58  GarageType     1379 non-null   object
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object
     64  GarageCond     1379 non-null   object
     65  PavedDrive     1460 non-null   object
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object
     73  Fence          281 non-null    object
     74  MiscFeature    54 non-null     object
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object
     79  SaleCondition  1460 non-null   object
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB


The first thing you notice is, there is a significant number of object type values in the data.  
Also, Alley, PoolQC, Fence and MiscFeature, have a considerable number of missing values.


```python
train_df.describe().T #This displays a statistical summary of numerical variables.
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1460.0</td>
      <td>730.500000</td>
      <td>421.610009</td>
      <td>1.0</td>
      <td>365.75</td>
      <td>730.5</td>
      <td>1095.25</td>
      <td>1460.0</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>1460.0</td>
      <td>56.897260</td>
      <td>42.300571</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>1201.0</td>
      <td>70.049958</td>
      <td>24.284752</td>
      <td>21.0</td>
      <td>59.00</td>
      <td>69.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>1460.0</td>
      <td>10516.828082</td>
      <td>9981.264932</td>
      <td>1300.0</td>
      <td>7553.50</td>
      <td>9478.5</td>
      <td>11601.50</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>1460.0</td>
      <td>6.099315</td>
      <td>1.382997</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>1460.0</td>
      <td>5.575342</td>
      <td>1.112799</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1460.0</td>
      <td>1971.267808</td>
      <td>30.202904</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1973.0</td>
      <td>2000.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1460.0</td>
      <td>1984.865753</td>
      <td>20.645407</td>
      <td>1950.0</td>
      <td>1967.00</td>
      <td>1994.0</td>
      <td>2004.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>1452.0</td>
      <td>103.685262</td>
      <td>181.066207</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>166.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1460.0</td>
      <td>443.639726</td>
      <td>456.098091</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>383.5</td>
      <td>712.25</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1460.0</td>
      <td>46.549315</td>
      <td>161.319273</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1460.0</td>
      <td>567.240411</td>
      <td>441.866955</td>
      <td>0.0</td>
      <td>223.00</td>
      <td>477.5</td>
      <td>808.00</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1460.0</td>
      <td>1057.429452</td>
      <td>438.705324</td>
      <td>0.0</td>
      <td>795.75</td>
      <td>991.5</td>
      <td>1298.25</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>1460.0</td>
      <td>1162.626712</td>
      <td>386.587738</td>
      <td>334.0</td>
      <td>882.00</td>
      <td>1087.0</td>
      <td>1391.25</td>
      <td>4692.0</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>1460.0</td>
      <td>346.992466</td>
      <td>436.528436</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>728.00</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>1460.0</td>
      <td>5.844521</td>
      <td>48.623081</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>572.0</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>1460.0</td>
      <td>1515.463699</td>
      <td>525.480383</td>
      <td>334.0</td>
      <td>1129.50</td>
      <td>1464.0</td>
      <td>1776.75</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>1460.0</td>
      <td>0.425342</td>
      <td>0.518911</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>1460.0</td>
      <td>0.057534</td>
      <td>0.238753</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>1460.0</td>
      <td>1.565068</td>
      <td>0.550916</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>1460.0</td>
      <td>0.382877</td>
      <td>0.502885</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>1460.0</td>
      <td>2.866438</td>
      <td>0.815778</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>1460.0</td>
      <td>1.046575</td>
      <td>0.220338</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>1460.0</td>
      <td>6.517808</td>
      <td>1.625393</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>1460.0</td>
      <td>0.613014</td>
      <td>0.644666</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>1379.0</td>
      <td>1978.506164</td>
      <td>24.689725</td>
      <td>1900.0</td>
      <td>1961.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1460.0</td>
      <td>1.767123</td>
      <td>0.747315</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1460.0</td>
      <td>472.980137</td>
      <td>213.804841</td>
      <td>0.0</td>
      <td>334.50</td>
      <td>480.0</td>
      <td>576.00</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>1460.0</td>
      <td>94.244521</td>
      <td>125.338794</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>857.0</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>1460.0</td>
      <td>46.660274</td>
      <td>66.256028</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>25.0</td>
      <td>68.00</td>
      <td>547.0</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>1460.0</td>
      <td>21.954110</td>
      <td>61.119149</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>552.0</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>1460.0</td>
      <td>3.409589</td>
      <td>29.317331</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>1460.0</td>
      <td>15.060959</td>
      <td>55.757415</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>1460.0</td>
      <td>2.758904</td>
      <td>40.177307</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>738.0</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>1460.0</td>
      <td>43.489041</td>
      <td>496.123024</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15500.0</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>1460.0</td>
      <td>6.321918</td>
      <td>2.703626</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>1460.0</td>
      <td>2007.815753</td>
      <td>1.328095</td>
      <td>2006.0</td>
      <td>2007.00</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>1460.0</td>
      <td>180921.195890</td>
      <td>79442.502883</td>
      <td>34900.0</td>
      <td>129975.00</td>
      <td>163000.0</td>
      <td>214000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>



Just from looking at this we can make quick simple basic inferences such as, half the number of houses were built in 1973 or earlier; the mean over all quality of houses sold is 6; there are no duplexes in 75% of houses sold; first remodelling was in 1950. For the target variable, minimum value of house sold is 34,900 dollars with maximum value of 755,000 dollars and mean of 163,000; A quarter of the houses are sold at about 130,000 or lower. Go ahead, make some conjectures of your own.


```python
# lets see what we can find from our object type variables
train_df.describe(include = [np.object])
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
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>KitchenQual</th>
      <th>Functional</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460</td>
      <td>1460</td>
      <td>91</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1452</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1423</td>
      <td>1423</td>
      <td>1422</td>
      <td>1423</td>
      <td>1422</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1459</td>
      <td>1460</td>
      <td>1460</td>
      <td>770</td>
      <td>1379</td>
      <td>1379</td>
      <td>1379</td>
      <td>1379</td>
      <td>1460</td>
      <td>7</td>
      <td>281</td>
      <td>54</td>
      <td>1460</td>
      <td>1460</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>25</td>
      <td>9</td>
      <td>8</td>
      <td>5</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>15</td>
      <td>16</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>top</th>
      <td>RL</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>Gd</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1151</td>
      <td>1454</td>
      <td>50</td>
      <td>925</td>
      <td>1311</td>
      <td>1459</td>
      <td>1052</td>
      <td>1382</td>
      <td>225</td>
      <td>1260</td>
      <td>1445</td>
      <td>1220</td>
      <td>726</td>
      <td>1141</td>
      <td>1434</td>
      <td>515</td>
      <td>504</td>
      <td>864</td>
      <td>906</td>
      <td>1282</td>
      <td>647</td>
      <td>649</td>
      <td>1311</td>
      <td>953</td>
      <td>430</td>
      <td>1256</td>
      <td>1428</td>
      <td>741</td>
      <td>1365</td>
      <td>1334</td>
      <td>735</td>
      <td>1360</td>
      <td>380</td>
      <td>870</td>
      <td>605</td>
      <td>1311</td>
      <td>1326</td>
      <td>1340</td>
      <td>3</td>
      <td>157</td>
      <td>49</td>
      <td>1267</td>
      <td>1198</td>
    </tr>
  </tbody>
</table>
</div>



Unlike describe method for numerical values, this only shows count (total number of instances), unique (number of unique values), top (most occurring attribute) and frequency (Frequency of top values). You can deduce observations  such as; 99.5% of streets are paved, as should be expected; 79% of houses sold are in low population density residential areas (RL) and that 51% of houses bought had excellent heating quality (HeatingQC)of gas forced air and so on.  
Next step is to push your analysis further by visualizing sections of the dtypes to get more and better insights, especially to find trends and patterns that may indicate a relationship.


```python
#first let us find the most correlated of attributes to our target variable since we cannot possibly plot all features.
corr = train_df.corr()
corr["SalePrice"].sort_values(ascending=False)
```




    SalePrice        1.000000
    OverallQual      0.790982
    GrLivArea        0.708624
    GarageCars       0.640409
    GarageArea       0.623431
    TotalBsmtSF      0.613581
    1stFlrSF         0.605852
    FullBath         0.560664
    TotRmsAbvGrd     0.533723
    YearBuilt        0.522897
    YearRemodAdd     0.507101
    GarageYrBlt      0.486362
    MasVnrArea       0.477493
    Fireplaces       0.466929
    BsmtFinSF1       0.386420
    LotFrontage      0.351799
    WoodDeckSF       0.324413
    2ndFlrSF         0.319334
    OpenPorchSF      0.315856
    HalfBath         0.284108
    LotArea          0.263843
    BsmtFullBath     0.227122
    BsmtUnfSF        0.214479
    BedroomAbvGr     0.168213
    ScreenPorch      0.111447
    PoolArea         0.092404
    MoSold           0.046432
    3SsnPorch        0.044584
    BsmtFinSF2      -0.011378
    BsmtHalfBath    -0.016844
    MiscVal         -0.021190
    Id              -0.021917
    LowQualFinSF    -0.025606
    YrSold          -0.028923
    OverallCond     -0.077856
    MSSubClass      -0.084284
    EnclosedPorch   -0.128578
    KitchenAbvGr    -0.135907
    Name: SalePrice, dtype: float64



Well, there you have it. These are the most positively and negatively correlated attributes with respect to price of houses in Ames. The negative correlations are very close to zero; as such, ineligible.  We can ignore them for now. Let us go ahead and visualize some of pos_corr to show if there are any linear relationships present.



```python
sns.set(palette="deep", font_scale=1.0)        

select_corr = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea",
               "TotalBsmtSF", "1stFlrSF","FullBath", "TotRmsAbvGrd", "YearBuilt",
               "YearRemodAdd","GarageYrBlt", "EnclosedPorch", "KitchenAbvGr", "MSSubClass", "LowQualFinSF"]

fig, ax = plt.subplots(5, 3, figsize=(20, 35))
for var, subplot in zip(select_corr, ax.flatten()):
    sns.scatterplot(x=var, y='SalePrice' , hue='MSZoning', data=train_df, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)

```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_16_0.png">


First thing you can notice is that most of the plots show positive linear relationships. There is also a large disparity in values between some independent variables and the target variable. This needs to be taken care of at a later stage. Next, as expected, we can see that when overall quality increases, the price goes up; Most of the houses are located at the RL zoning classification; Houses in commercial areas (C) are the cheapest as they seem to have lowest overall quality; Houses in low residential density areas (RL) have higher above ground living area, larger garage in size and number of cars it can contain and boasts of the most expensive houses; Houses in RM & C zoning classification have the smallest average of total basement area; Highest grade bathrooms starts at approximately $180,000; only one instance of 14 rooms in the training dataset; and so on.
It is also quite evident that some of the variables are not uniformly distributed.
Next we take a look at the relationship between some randomly selected categorical variables and the target variable.




```python
sns.set(palette="deep", font_scale=1.0)        

select_var = [
  'MSZoning', 'LotShape', 'Neighborhood', 'Condition2', 'SaleCondition', 'BsmtCond', 'ExterCond',
    "MasVnrType", "Condition1", "KitchenQual", 'BldgType', 'Functional'
]
fig, ax = plt.subplots(4, 3, figsize=(20, 35))
for var, subplot in zip(select_var, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=train_df, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)

```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_18_0.png">


Boxplots are a great way to give a clear and precise overview of our categorical values. Overall, notice that most of the plots have a fairly normal distribution, however, there are lots of outliers, with high standard deviation and variance present. The box plots are also short (except Neighborhood) which implies that the data points are similar and in short range. From the first plot, the zoning category with the highest price average is the floating village residential; the neighborhood with the most expensive houses are Northridge Heights, Northridge and Stone Brook; From condition1, proximity to the park (PosN) is positively correlated to price of house; In exterCond, excellent exterior materials are more dispersed after 150,000; Excellent kitchen qualities have an interquartile range of approximately 250,000 and 400,000.   


```python
#let us look at neighborhood more closely
plt.figure(figsize=(9,5))
sns.stripplot(x = train_df.Neighborhood, y = train_df.SalePrice,
              order = np.sort(train_df.Neighborhood.unique()),
              alpha=1)

plt.xticks(rotation=45)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_20_0.png">



```python
#Neughborhood vs Overall Quality
plt.figure(figsize=(9,5))
sns.pointplot(x = train_df.Neighborhood, y = train_df.OverallQual)
plt.xticks(rotation=45)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_21_0.png">



```python
#compare SalePrice and SaleType with respect to SaleCondition
plt.figure(figsize=(9,5))
sns.scatterplot(x="SaleType",y="OverallQual",
                hue="SaleCondition",data=train_df)
```







<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_22_1.png">



```python
#What are the number of houses sold per year
year_count = train_df.YrSold.value_counts()
plt.figure(figsize= (9, 5))
sns.barplot(year_count.index, year_count.values, alpha=1)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_23_0.png">


## 4. Prepare Data for Machine Learning Algorithms.
This stage involves data cleaning, wrangling and feature engineering. This prepares the data in a the best possible way that it can be accepted by ML models.

### 4.1.  General Tidying


```python
data = pd.concat([train_df, test_df])

data["MSSubClass"] = data["MSSubClass"].astype(str) #Find variables in wrong types. MSSubClass is ordinal, and not an int.

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2919 entries, 0 to 1458
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             2919 non-null   int64  
     1   MSSubClass     2919 non-null   object
     2   MSZoning       2915 non-null   object
     3   LotFrontage    2433 non-null   float64
     4   LotArea        2919 non-null   int64  
     5   Street         2919 non-null   object
     6   Alley          198 non-null    object
     7   LotShape       2919 non-null   object
     8   LandContour    2919 non-null   object
     9   Utilities      2917 non-null   object
     10  LotConfig      2919 non-null   object
     11  LandSlope      2919 non-null   object
     12  Neighborhood   2919 non-null   object
     13  Condition1     2919 non-null   object
     14  Condition2     2919 non-null   object
     15  BldgType       2919 non-null   object
     16  HouseStyle     2919 non-null   object
     17  OverallQual    2919 non-null   int64  
     18  OverallCond    2919 non-null   int64  
     19  YearBuilt      2919 non-null   int64  
     20  YearRemodAdd   2919 non-null   int64  
     21  RoofStyle      2919 non-null   object
     22  RoofMatl       2919 non-null   object
     23  Exterior1st    2918 non-null   object
     24  Exterior2nd    2918 non-null   object
     25  MasVnrType     2895 non-null   object
     26  MasVnrArea     2896 non-null   float64
     27  ExterQual      2919 non-null   object
     28  ExterCond      2919 non-null   object
     29  Foundation     2919 non-null   object
     30  BsmtQual       2838 non-null   object
     31  BsmtCond       2837 non-null   object
     32  BsmtExposure   2837 non-null   object
     33  BsmtFinType1   2840 non-null   object
     34  BsmtFinSF1     2918 non-null   float64
     35  BsmtFinType2   2839 non-null   object
     36  BsmtFinSF2     2918 non-null   float64
     37  BsmtUnfSF      2918 non-null   float64
     38  TotalBsmtSF    2918 non-null   float64
     39  Heating        2919 non-null   object
     40  HeatingQC      2919 non-null   object
     41  CentralAir     2919 non-null   object
     42  Electrical     2918 non-null   object
     43  1stFlrSF       2919 non-null   int64  
     44  2ndFlrSF       2919 non-null   int64  
     45  LowQualFinSF   2919 non-null   int64  
     46  GrLivArea      2919 non-null   int64  
     47  BsmtFullBath   2917 non-null   float64
     48  BsmtHalfBath   2917 non-null   float64
     49  FullBath       2919 non-null   int64  
     50  HalfBath       2919 non-null   int64  
     51  BedroomAbvGr   2919 non-null   int64  
     52  KitchenAbvGr   2919 non-null   int64  
     53  KitchenQual    2918 non-null   object
     54  TotRmsAbvGrd   2919 non-null   int64  
     55  Functional     2917 non-null   object
     56  Fireplaces     2919 non-null   int64  
     57  FireplaceQu    1499 non-null   object
     58  GarageType     2762 non-null   object
     59  GarageYrBlt    2760 non-null   float64
     60  GarageFinish   2760 non-null   object
     61  GarageCars     2918 non-null   float64
     62  GarageArea     2918 non-null   float64
     63  GarageQual     2760 non-null   object
     64  GarageCond     2760 non-null   object
     65  PavedDrive     2919 non-null   object
     66  WoodDeckSF     2919 non-null   int64  
     67  OpenPorchSF    2919 non-null   int64  
     68  EnclosedPorch  2919 non-null   int64  
     69  3SsnPorch      2919 non-null   int64  
     70  ScreenPorch    2919 non-null   int64  
     71  PoolArea       2919 non-null   int64  
     72  PoolQC         10 non-null     object
     73  Fence          571 non-null    object
     74  MiscFeature    105 non-null    object
     75  MiscVal        2919 non-null   int64  
     76  MoSold         2919 non-null   int64  
     77  YrSold         2919 non-null   int64  
     78  SaleType       2918 non-null   object
     79  SaleCondition  2919 non-null   object
     80  SalePrice      1460 non-null   float64
    dtypes: float64(12), int64(25), object(44)
    memory usage: 1.8+ MB


### 4.2. Feature Extraction


```python
#Try out attribute combination
data["OverAll"] = data["OverallQual"] / data["OverallCond"]


data["AgeSold"] = data["YrSold"] - data["YearBuilt"]


data["TotalBsmt"] = data["BsmtFinSF2"] + data["BsmtFinSF1"]


data["TotalBath"] = data["FullBath"] + data["HalfBath"] *0.5

```


```python
data["OverAll"] = data["OverallQual"] / data["OverallCond"]

attr_combo1 = ["OverAll", "OverallQual", "OverallCond"]
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
for var, subplot in zip(attr_combo1, ax.flatten()):
    sns.scatterplot(x=var, y='SalePrice' , hue="ExterQual", data=data, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_29_0.png">


### 4.3. Checking for missing values and Dropping irrelevant Faetures


```python
#First lets see how many missing values per attribute
null = data.isnull().sum().sort_values(ascending=False)
null[null>0]
```




    PoolQC          2909
    MiscFeature     2814
    Alley           2721
    Fence           2348
    SalePrice       1459
    FireplaceQu     1420
    LotFrontage      486
    GarageQual       159
    GarageYrBlt      159
    GarageFinish     159
    GarageCond       159
    GarageType       157
    BsmtCond          82
    BsmtExposure      82
    BsmtQual          81
    BsmtFinType2      80
    BsmtFinType1      79
    MasVnrType        24
    MasVnrArea        23
    MSZoning           4
    Functional         2
    BsmtFullBath       2
    Utilities          2
    BsmtHalfBath       2
    KitchenQual        1
    Exterior2nd        1
    TotalBsmt          1
    Exterior1st        1
    Electrical         1
    GarageCars         1
    GarageArea         1
    BsmtUnfSF          1
    BsmtFinSF2         1
    BsmtFinSF1         1
    SaleType           1
    TotalBsmtSF        1
    dtype: int64




```python
def corr_(data):
    correlation = data.corr()

    fig, ax = plt.subplots(figsize=(30,20))
    sns.heatmap(correlation, vmax=1.0, center=0, fmt='.2f', square=True,
               linewidth=.5, annot=True, cbar_kws={'shrink': .70})

    plt.show()

corr_(data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kag/output_32_0.png">



```python
# I think the features with high missing values are unavailable not missing (Note the differece). Eg, there are no pools in
# almost all the houses. For this reason, we will not drop any variable with missing instances but fill them. However, we will
# drop multicollinear predictors with a threashold of [-7,7].

data = data.drop(['Id', 'SalePrice'], axis=1).copy()
data = data.drop(['1stFlrSF', 'TotalBsmt', 'FullBath', 'OverAll', 'AgeSold'], axis=1) #from correlation
data = data.drop(['Utilities', 'Street', 'PoolQC', 'MiscFeature', 'HalfBath', 'LowQualFinSF', 'GarageQual', '3SsnPorch'], axis=1)

```

## 4.4. Feature Transform


```python
train = data.iloc[:1460]
test = data.iloc[1460:]
```


```python
train.sample(5)
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>Fence</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>TotalBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>694</th>
      <td>50</td>
      <td>RM</td>
      <td>51.0</td>
      <td>6120</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>BrkSide</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>6</td>
      <td>1936</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Fa</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>927.0</td>
      <td>927.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>472</td>
      <td>1539</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1995.0</td>
      <td>Unf</td>
      <td>2.0</td>
      <td>576.0</td>
      <td>TA</td>
      <td>Y</td>
      <td>112</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>MnPrv</td>
      <td>0</td>
      <td>4</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>908</th>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>8885</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1983</td>
      <td>1983</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>BLQ</td>
      <td>301.0</td>
      <td>ALQ</td>
      <td>324.0</td>
      <td>239.0</td>
      <td>864.0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>902</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1983.0</td>
      <td>Unf</td>
      <td>2.0</td>
      <td>484.0</td>
      <td>TA</td>
      <td>Y</td>
      <td>164</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>MnPrv</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>20</td>
      <td>FV</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>No</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>1293.0</td>
      <td>1293.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>1301</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2006.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>572.0</td>
      <td>TA</td>
      <td>Y</td>
      <td>216</td>
      <td>121</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2006</td>
      <td>New</td>
      <td>Partial</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>341</th>
      <td>20</td>
      <td>RH</td>
      <td>60.0</td>
      <td>8400</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>SawyerW</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>4</td>
      <td>1950</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>AsbShng</td>
      <td>None</td>
      <td>0.0</td>
      <td>Fa</td>
      <td>Fa</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>Fa</td>
      <td>No</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>721.0</td>
      <td>721.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>841</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>CarPort</td>
      <td>1950.0</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>294.0</td>
      <td>TA</td>
      <td>N</td>
      <td>250</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>796</th>
      <td>20</td>
      <td>RL</td>
      <td>71.0</td>
      <td>8197</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Sawyer</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>1977</td>
      <td>1977</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>BrkFace</td>
      <td>148.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>660.0</td>
      <td>660.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>1285</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1977.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>528.0</td>
      <td>TA</td>
      <td>Y</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>MnPrv</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_num = train.select_dtypes(include=[np.number])#select all numeric columns
train_num.head()
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
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>2ndFlrSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>TotalBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>856.0</td>
      <td>854</td>
      <td>1710</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2003.0</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>284.0</td>
      <td>1262.0</td>
      <td>0</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1976.0</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>920.0</td>
      <td>866</td>
      <td>1786</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2001.0</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>756.0</td>
      <td>756</td>
      <td>1717</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1998.0</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1145.0</td>
      <td>1053</td>
      <td>2198</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>2000.0</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.pipeline import Pipeline as pl
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import RobustScaler as rs

num_pipeline = pl([
    ('imputer', si(strategy="mean")),
    ('scaler', rs()),
    ])

train_num_trx = num_pipeline.fit_transform(train_num)
print(train_num_trx)
print('***' * 30)
print(train_num_trx.shape)
print('***' * 30)
print(np.isnan(train_num_trx).sum())
```

    [[-0.26578728 -0.25407609  0.5        ... -1.33333333  0.
       0.33333333]
     [ 0.5236864   0.03001482  0.         ... -0.33333333 -0.5
       0.        ]
     [-0.10789255  0.43762352  0.5        ...  1.          0.
       0.33333333]
     ...
     [-0.2131557  -0.10783103  0.5        ... -0.33333333  1.
       0.        ]
     [-0.10789255  0.05891798 -0.5        ... -0.66666667  1.
      -0.66666667]
     [ 0.26052851  0.11326581 -0.5        ...  0.          0.
      -0.33333333]]
    ******************************************************************************************
    (1460, 31)
    ******************************************************************************************
    0


## Non numeric Pipeline



```python
train_ordinal = train.select_dtypes(exclude=[np.number])
train_ordinal.head()
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>KitchenQual</th>
      <th>Functional</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>Fence</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>RL</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Gd</td>
      <td>Typ</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
 import category_encoders as ce

ordinal_pipeline = pl([
    ('imputer', si(strategy="most_frequent", fill_value='None')),
    ('ord_encoder', ce.OrdinalEncoder(return_df=False)),
    ('scaler', rs()),
])

train_ord_trx = ordinal_pipeline.fit_transform(train_ordinal)  
print(train_ord_trx)
print('***' * 30)
print(train_ord_trx.shape)
print('***' * 30)
print(np.isnan(train_ord_trx).sum())
```

    [[ 0.2  0.   0.  ...  0.   0.   0. ]
     [-0.6  0.   0.  ...  0.   0.   0. ]
     [ 0.2  0.   0.  ...  0.   0.   0. ]
     ...
     [ 0.4  0.   0.  ...  2.   0.   0. ]
     [-0.6  0.   0.  ...  0.   0.   0. ]
     [-0.6  0.   0.  ...  0.   0.   0. ]]
    ******************************************************************************************
    (1460, 39)
    ******************************************************************************************
    0



```python
#combine numerical and categorical pipelines
from sklearn.compose import ColumnTransformer as colt

num_attribs = list(train_num)
#nom_attribs = list(train_nominal)
ord_attribs = list(train_ordinal)


full_pipeline = colt([
    ("num", num_pipeline, num_attribs),
   # ("nom", nominal_pipeline, nom_attribs),
    ("ord", ordinal_pipeline, ord_attribs)
])
```


```python
X_train = full_pipeline.fit_transform(train)
X_test = full_pipeline.transform(test)
```

### Recheck and Rename


```python
print(X_train.shape)
print(X_test.shape)
```

    (1460, 70)
    (1459, 70)



```python
#To reduce the distance between values, such that a slght change in x affects y.
y = np.log1p(train_df['SalePrice']).copy()
y_test = np.log1p(sub_df['SalePrice']).copy()
```

### 5.0. Select and Train a Model


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
```


```python
linear = LinearRegression()
linear.fit(X_train, y)
linear_predict = linear.predict(X_test)
linear_predict
```




    array([11.61963892, 11.93880308, 12.04341483, ..., 12.05827928,
           11.69047332, 12.42217636])




```python
#We can see the actual values and make visual comparisons

print("Top 3 Labels:", list(y_test.head(3)))
print("Last 3 Labels:", list(y_test.tail(3)))
```

    Top 3 Labels: [12.039297922968691, 12.142916602935376, 12.120431330864875]
    Last 3 Labels: [12.297846686923938, 12.127707128738267, 12.142828575069593]



```python
lr_mse = mse(linear_predict, y_test)
lr_rmse = np.sqrt(lr_mse)
lr_rmse
```




    0.37529958427243454




```python
forest = RandomForestRegressor(n_estimators=500, random_state=42)
forest.fit(X_train, y)
forest_predict = forest.predict(X_test)
forest_predict
```




    array([11.73574016, 11.94428311, 12.08468668, ..., 11.94771477,
           11.66495922, 12.37360112])




```python
rfr_mse = mse(forest_predict, y_test)
rfr_rmse = np.sqrt(rfr_mse)
rfr_rmse
```




    0.3639948694315479




```python
lasso = Lasso(max_iter=1000, alpha=0.01)
lasso.fit(X_train, y)
lasso_pred = lasso.predict(X_test)
lasso_pred
```




    array([11.73449396, 11.88775546, 12.05602704, ..., 12.02604719,
           11.6888728 , 12.38120214])




```python
lasso_mse = mse(lasso_pred, y_test)
lasso_rmse = np.sqrt(lasso_mse)
lasso_rmse
```




    0.3458549542156051




```python
train_matrix = xgb.DMatrix(X_train, y)
test_matrix = xgb.DMatrix(X_test)

xg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
         random_state=42)
xg .fit(X_train, y)
xg_pred = xg.predict(X_test)
xg_pred
```




    array([11.706657, 11.984388, 12.152975, ..., 12.082707, 11.706811,
           12.328882], dtype=float32)




```python
xg_mse = mse(xg_pred, y_test)
xg_rmse = np.sqrt(xg_mse)
xg_rmse
```




    0.3848714208977196




```python
ridge = Ridge()
ridge.fit(X_train, y)
ridge_pred = ridge.predict(X_test)
ridge_pred
```




    array([11.62101732, 11.94041983, 12.0437113 , ..., 12.05949654,
           11.69001701, 12.42200935])




```python
ridge_mse = mse(ridge_pred, y_test)
ridge_rmse = np.sqrt(ridge_mse)
ridge_rmse
```




    0.37495026973019624




```python
svm = SVR()
svm.fit(X_train, y)
svm_pred = svm.predict(X_test)
svm_pred
```




    array([11.97421201, 12.25842922, 12.0408935 , ..., 12.04879778,
           11.87755258, 12.05781983])




```python
svm_mse = mse(svm_pred, y_test)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
```




    0.1559147202143059




```python
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y)
knn_pred = knn.predict(X_test)
knn_pred
```




    array([11.84492742, 11.66607959, 12.13417428, ..., 11.86700169,
           11.9627464 , 12.30373907])




```python
knn_mse = mse(knn_pred, y_test)
knn_rmse = np.sqrt(knn_mse)
knn_rmse
```




    0.32369617565934844



Our support vector regressor outperforms the others by a mile. It is efinitely overfitting. For practice, fine-tune the other regressors and compare how they perform.

## Validating & Fine-Tuning Model

We now have some promising models--random forest and xgboost. Next step is to fine-tune and validate them. Notice in the models above, we have placed the barest of parameters. This is beacuse there is no human way to exaustively find the best combinations of hyperparameters for our choosen model. Think of it as tuning a radio nob to find the clearest station. Thanks to scikitlearn's GridSearchCV and RandomisedSearchCV, we can do this automatically.  


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

xgb_regg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

params_xgb = {
        'max_depth': [3, 4, 5, 6, 7],
        'num_boost_round': [10, 25],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6],
        'colsample_bylevel': [0.4, 0.5, 0.6],
        'min_child_weight': [1.0, 3.0],
        'gamma': [0, 0.25],
       'reg_lambda': [1.0, 5.0, 7.0],
        'n_estimators': randint(200, 2000)
        }

random_ = RandomizedSearchCV(xgb_regg, param_distributions=params_xgb, n_iter=20, cv=5,
         scoring='neg_mean_squared_error', n_jobs=1,  random_state=42)
```


```python
random_.fit(X_train, y)
```




    RandomizedSearchCV(cv=5, error_score=nan,
                       estimator=XGBRegressor(base_score=0.5, booster='gbtree',
                                              colsample_bylevel=1,
                                              colsample_bynode=1,
                                              colsample_bytree=1, gamma=0,
                                              importance_type='gain',
                                              learning_rate=0.1, max_delta_step=0,
                                              max_depth=3, min_child_weight=1,
                                              missing=None, n_estimators=100,
                                              n_jobs=1, nthread=None,
                                              objective='reg:squarederror',
                                              random_state=42, reg...
                                            'gamma': [0, 0.25],
                                            'max_depth': [3, 4, 5, 6, 7],
                                            'min_child_weight': [1.0, 3.0],
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x0000022AC8AB1748>,
                                            'num_boost_round': [10, 25],
                                            'reg_lambda': [1.0, 5.0, 7.0],
                                            'subsample': [0.9, 1.0]},
                       pre_dispatch='2*n_jobs', random_state=42, refit=True,
                       return_train_score=False, scoring='neg_mean_squared_error',
                       verbose=0)




```python
random_.best_params_
```




    {'colsample_bylevel': 0.4,
     'colsample_bytree': 0.4,
     'gamma': 0,
     'max_depth': 3,
     'min_child_weight': 3.0,
     'n_estimators': 1352,
     'num_boost_round': 25,
     'reg_lambda': 7.0,
     'subsample': 0.9}




```python
 random_.best_estimator_
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.4,
                 colsample_bynode=1, colsample_bytree=0.4, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=3.0, missing=None, n_estimators=1352,
                 n_jobs=1, nthread=None, num_boost_round=25,
                 objective='reg:squarederror', random_state=42, reg_alpha=0,
                 reg_lambda=7.0, scale_pos_weight=1, seed=None, silent=None,
                 subsample=0.9, verbosity=1)




```python
model = random_.best_estimator_

pred2 = model.predict(X_test)
```


```python
prediction2 = np.expm1(pred2)
output2 = pd.DataFrame({'Id':test_df.Id, 'SalePrice':prediction2})
output2.head()
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
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>118886.632812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>161320.671875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>187420.640625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>192596.515625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>181116.140625</td>
    </tr>
  </tbody>
</table>
</div>




```python
output2.to_csv('xgb8.csv', index=False)
```


```python
feature_importance = random_.best_estimator_.feature_importances_
attributes = num_attribs + ord_attribs
sorted(zip(feature_importance, attributes), reverse=True)

```




    [(0.17045057, 'OverallQual'),
     (0.15516618, 'GarageCars'),
     (0.07165706, 'Fireplaces'),
     (0.047740616, 'TotalBath'),
     (0.044116538, 'CentralAir'),
     (0.042846948, 'GrLivArea'),
     (0.029083796, 'YearRemodAdd'),
     (0.026234755, 'GarageArea'),
     (0.026158102, 'YearBuilt'),
     (0.025795698, 'TotalBsmtSF'),
     (0.02197889, 'OverallCond'),
     (0.020921756, 'GarageYrBlt'),
     (0.016097259, 'BsmtFinSF1'),
     (0.016042368, 'BsmtFinType1'),
     (0.015377405, 'KitchenAbvGr'),
     (0.014440038, 'Functional'),
     (0.013417843, 'GarageFinish'),
     (0.013118019, 'MSZoning'),
     (0.012961783, '2ndFlrSF'),
     (0.011720444, 'Condition2'),
     (0.011579358, 'BsmtQual'),
     (0.0094911475, 'BsmtExposure'),
     (0.008285343, 'BldgType'),
     (0.007862613, 'Foundation'),
     (0.0077409633, 'KitchenQual'),
     (0.007587692, 'LotArea'),
     (0.0071345335, 'PavedDrive'),
     (0.0067442725, 'LotFrontage'),
     (0.0063789533, 'BsmtFullBath'),
     (0.006071903, 'Condition1'),
     (0.00599091, 'Fence'),
     (0.0058421427, 'TotRmsAbvGrd'),
     (0.0053032683, 'MiscVal'),
     (0.0052680154, 'Neighborhood'),
     (0.0052600936, 'PoolArea'),
     (0.0052061058, 'ScreenPorch'),
     (0.0051219543, 'SaleCondition'),
     (0.0048817596, 'OpenPorchSF'),
     (0.004331854, 'ExterQual'),
     (0.0039200904, 'RoofMatl'),
     (0.003806459, 'ExterCond'),
     (0.0036296323, 'LotShape'),
     (0.0036123004, 'WoodDeckSF'),
     (0.0034323013, 'BsmtFinSF2'),
     (0.003272554, 'HeatingQC'),
     (0.0032045993, 'Exterior1st'),
     (0.0030321914, 'BedroomAbvGr'),
     (0.003013601, 'HouseStyle'),
     (0.0029762562, 'GarageType'),
     (0.0027719024, 'MSSubClass'),
     (0.0027711922, 'MoSold'),
     (0.002718122, 'SaleType'),
     (0.002393617, 'GarageCond'),
     (0.0023310797, 'Exterior2nd'),
     (0.0023181213, 'YrSold'),
     (0.002283722, 'BsmtFinType2'),
     (0.002270854, 'BsmtCond'),
     (0.0021721106, 'EnclosedPorch'),
     (0.0021401297, 'MasVnrType'),
     (0.0021189386, 'MasVnrArea'),
     (0.0020685012, 'LotConfig'),
     (0.0020539796, 'FireplaceQu'),
     (0.0020457623, 'Heating'),
     (0.0020246482, 'BsmtUnfSF'),
     (0.0019086191, 'RoofStyle'),
     (0.001887537, 'Electrical'),
     (0.0018400064, 'LandSlope'),
     (0.0017197807, 'LandContour'),
     (0.001689854, 'Alley'),
     (0.0011625615, 'BsmtHalfBath')]



<img src="{{ site.url }}{{ site.baseurl }}/images/kag/hp.png">  


```python

```
