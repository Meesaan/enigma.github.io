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

Head to [Kaggle.com](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) to get a full description of this dataset.  


## 1. Generate an idea  
    Firstly, it is always good practice to have knowledge of what the objective of the project is. This means analysing who/what/where/when stands to benefit from your outcome. It helps you understand as well as create a guideline for the problem on a personal scale.  
    For this project, let us assume we are working for  real estate investors who would like to predict the SalePrice of houses in Ames, Iowa, given 80 factors or predictor variables. They would like to know if investing would be a good business idea. With this in mind, you can start thinking of types of information you could provide your employer that would help them gain competitive advantage. How would you deliver your findings to a non-science audience? You should develop these thought processes from the begining.
    At this point, if you have previewed the data, you can tell that this is a batch univariate regression problem. Although RMSE has been choosen as its performance measure, note that if there are lots of outliers, you could also try MAE.  

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
      <td>491</td>
      <td>492</td>
      <td>50</td>
      <td>RL</td>
      <td>79.0</td>
      <td>9490</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>6</td>
      <td>7</td>
      <td>1941</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>BLQ</td>
      <td>403</td>
      <td>Rec</td>
      <td>165</td>
      <td>238</td>
      <td>806</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>958</td>
      <td>620</td>
      <td>0</td>
      <td>1578</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Fa</td>
      <td>5</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1941.0</td>
      <td>Unf</td>
      <td>1</td>
      <td>240</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>133000</td>
    </tr>
    <tr>
      <td>463</td>
      <td>464</td>
      <td>70</td>
      <td>RL</td>
      <td>74.0</td>
      <td>11988</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>7</td>
      <td>1934</td>
      <td>1995</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Stucco</td>
      <td>Stucco</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>LwQ</td>
      <td>326</td>
      <td>Unf</td>
      <td>0</td>
      <td>389</td>
      <td>715</td>
      <td>GasA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>849</td>
      <td>811</td>
      <td>0</td>
      <td>1660</td>
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
      <td>Gd</td>
      <td>Detchd</td>
      <td>1939.0</td>
      <td>Unf</td>
      <td>1</td>
      <td>240</td>
      <td>TA</td>
      <td>TA</td>
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
      <td>8</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188700</td>
    </tr>
    <tr>
      <td>937</td>
      <td>938</td>
      <td>60</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9675</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Somerst</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
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
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>341</td>
      <td>Unf</td>
      <td>0</td>
      <td>772</td>
      <td>1113</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1113</td>
      <td>858</td>
      <td>0</td>
      <td>1971</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2005.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>689</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>253000</td>
    </tr>
    <tr>
      <td>346</td>
      <td>347</td>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>12772</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1960</td>
      <td>1998</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>Mn</td>
      <td>BLQ</td>
      <td>498</td>
      <td>Unf</td>
      <td>0</td>
      <td>460</td>
      <td>958</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>958</td>
      <td>0</td>
      <td>0</td>
      <td>958</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1960.0</td>
      <td>RFn</td>
      <td>1</td>
      <td>301</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>15500</td>
      <td>4</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>151500</td>
    </tr>
    <tr>
      <td>1266</td>
      <td>1267</td>
      <td>190</td>
      <td>RM</td>
      <td>60.0</td>
      <td>10120</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>2fmCon</td>
      <td>2.5Unf</td>
      <td>7</td>
      <td>4</td>
      <td>1910</td>
      <td>1950</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>Fa</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>925</td>
      <td>925</td>
      <td>GasA</td>
      <td>TA</td>
      <td>N</td>
      <td>FuseF</td>
      <td>964</td>
      <td>925</td>
      <td>0</td>
      <td>1889</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1960.0</td>
      <td>Unf</td>
      <td>1</td>
      <td>308</td>
      <td>TA</td>
      <td>TA</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>264</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>122000</td>
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
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
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
      <td>Id</td>
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
      <td>MSSubClass</td>
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
      <td>LotFrontage</td>
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
      <td>LotArea</td>
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
      <td>OverallQual</td>
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
      <td>OverallCond</td>
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
      <td>YearBuilt</td>
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
      <td>YearRemodAdd</td>
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
      <td>MasVnrArea</td>
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
      <td>BsmtFinSF1</td>
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
      <td>BsmtFinSF2</td>
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
      <td>BsmtUnfSF</td>
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
      <td>TotalBsmtSF</td>
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
      <td>1stFlrSF</td>
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
      <td>2ndFlrSF</td>
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
      <td>LowQualFinSF</td>
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
      <td>GrLivArea</td>
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
      <td>BsmtFullBath</td>
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
      <td>BsmtHalfBath</td>
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
      <td>FullBath</td>
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
      <td>HalfBath</td>
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
      <td>BedroomAbvGr</td>
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
      <td>KitchenAbvGr</td>
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
      <td>TotRmsAbvGrd</td>
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
      <td>Fireplaces</td>
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
      <td>GarageYrBlt</td>
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
      <td>GarageCars</td>
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
      <td>GarageArea</td>
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
      <td>WoodDeckSF</td>
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
      <td>OpenPorchSF</td>
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
      <td>EnclosedPorch</td>
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
      <td>3SsnPorch</td>
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
      <td>ScreenPorch</td>
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
      <td>PoolArea</td>
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
      <td>MiscVal</td>
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
      <td>MoSold</td>
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
      <td>YrSold</td>
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
      <td>SalePrice</td>
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



Just from looking at this we can make quick simple basic inferences such as, half the number of houses were built in 1973 or earlier; the mean over all quality of houses sold is 6; there are no duplexes in 75% of houses sold; first remodeling was in 1950. For the target variable, minimum value of house sold is 34,900 dollars with maximum value of 755,000 dollars and mean of 163,000; A quarter of the houses are sold at about 130,000 or lower. Go ahead, make some conjectures of your own.


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
      <td>count</td>
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
      <td>unique</td>
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
      <td>top</td>
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
      <td>freq</td>
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



Unlike describe method for numerical values, this only shows count (total number of instances), unique (number of unque values), top (most occuring attribute) and frequency (Frequency of top values). You can deduce observations  such as; 99.5% of streets are paved, as should be expected; 79% of houses sold are in low population density residential areas (RL) and that 51% of houses bought had excellent heating quality (HeatingQC)of gas forced air and so on.  
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



Well, there you have it. These are the most positively and negatively correlated attributes with respect to price of houses in Ames. The neagtive correlations are very close to zero; as such, inegligible.  We can ignore them for now. Let us go ahead and visualize some of pos_corr to show if there are any linear relationships present.



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


![png](output_16_0.png)


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


![png](output_18_0.png)


Boxplots are a great way to give a clear and precise overview of our categorical values. Overall, notice that most of the plots have a fairly normal distribution, however, there are lots of outliers, with high standard deviation and variance present. The box plots are also short (except Neighborhood) which implies that the data points are similar and in short range. From the first plot, the zoning category with the highest price average is the floating village residential; the neighborhood with the most expensive houses are Northridge Heights, Northridge and Stone Brook; From condition1, proximity to the park (PosN) is positively correlated to price of house; In exterCond, excellent exterior materials are more dispersed after 150,000; Excellent kitchen qualities have an interquartile range of approximately 250,000 and 400,000.   


```python
#let us look at neighborhood more closely
plt.figure(figsize=(9,5))()
sns.stripplot(x = train_df.Neighborhood, y = train_df.SalePrice,
              order = np.sort(train_df.Neighborhood.unique()),
              alpha=1)

plt.xticks(rotation=45)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-769aafcefd30> in <module>
          1 #let us look at neighborhood more closely
    ----> 2 plt.figure(figsize=(9,5))()
          3 sns.stripplot(x = train_df.Neighborhood, y = train_df.SalePrice,
          4               order = np.sort(train_df.Neighborhood.unique()),
          5               alpha=1)


    TypeError: 'Figure' object is not callable



    <Figure size 648x360 with 0 Axes>



```python
#Neughborhood vs Overall Quality
plt.figure(figsize=(9,5))
sns.pointplot(x = train_df.Neighborhood, y = train_df.OverallQual)
plt.xticks(rotation=45)
```


```python
#compare SalePrice and SaleType with respect to SaleCondition
plt.figure(figsize=(9,5))
sns.scatterplot(x="SaleType",y="OverallQual",
                hue="SaleCondition",data=train_df)
```


```python
#What are the number of houses sold per year
year_count = train_df.YrSold.value_counts()
plt.figure(figsize= (9, 5))
sns.barplot(year_count.index, year_count.values, alpha=1)
plt.show()
```

## 4. Prepare Data for Machine Learning Algorithms.
This stage involves data cleaning, wrangling and feature engineering. This prepares the data in a the best possible way that it can be accepted by ML models.

### 4.1.  General Tidying


```python
data = pd.concat([train_df, test_df])

data["MSSubClass"] = data["MSSubClass"].astype(str) #Find variables in wrong types. MSSubClass is ordinal, and not an int.

data.info()
```

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

### 4.3. Checking for missing values and Dropping irrelevant Faetures


```python
#First lets see how many missing values per attribute
null = data.isnull().sum().sort_values(ascending=False)
null[null>0]
```


```python
def corr_(data):
    correlation = data.corr()

    fig, ax = plt.subplots(figsize=(30,20))
    sns.heatmap(correlation, vmax=1.0, center=0, fmt='.2f', square=True,
               linewidth=.5, annot=True, cbar_kws={'shrink': .70})

    plt.show()

corr_(data)
```


```python
# I think the features with high missing values are unavailable not missing (Note the differece). Eg, there are no pools in
# almost all the houses. For this reason, we will not drop any variable with missing instances but fill them. However, we will
# drop multicollinear predictors with a threashold of 7 such that: -0.7<=0<=0.7.

data = data.drop(['Id', 'SalePrice'], axis=1).copy()
data = data.drop(['1stFlrSF', 'TotalBsmt', 'FullBath', 'OverAll', 'AgeSold'], axis=1) #from correlation
data = data.drop(['Utilities', 'Street', 'PoolQC', 'MiscFeature'], axis=1)

```

## 4.4. Feature Transform


```python
train = data.iloc[:1460]
test = data.iloc[1460:]
```


```python
train.sample(5)
```


```python
train_num = train.select_dtypes(include=[np.number])#select all numeric columns
train_num.head()
```


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

## Non numeric Pipeline



```python
train_ordinal = train.select_dtypes(exclude=[np.number])
train_ordinal.head()
```


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


```python

```


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


```python
#To reduce the distance between values, such that a slght change in x affects y.
y = np.log1p(train_df['SalePrice']).copy()
y_test = np.log1p(sub_df['SalePrice']).copy()
```

### 5.0. Select and Train a Model


```python
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as knr

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
```


```python
linear = lr()
linear.fit(X_train, y)
train_predict_1 = linear.predict(X_test)
train_predict_1
```


```python
#We can see the actual values and make visual comparisons

print("Top 3 Labels:", list(y_test.head(3)))
print("Last 3 Labels:", list(y_test.tail(3)))
```


```python
forest = rfr(n_estimators=500, random_state=42)
forest.fit(X_train, y)
train_predict_2 = forest.predict(X_test)
train_predict_2
```


```python
lasso = Lasso(max_iter=1000, alpha=0.01)
lasso.fit(X_train, y)
train_predict_3 = lasso.predict(X_test)
train_predict_3
```


```python
train_matrix = xgb.DMatrix(X_train, y)
test_matrix = xgb.DMatrix(X_test)

xg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
         random_state=42)
xg .fit(X_train, y)
train_predict_4 = xg.predict(X_test)
train_predict_4
```


```python
ridge = Ridge()
ridge.fit(X_train, y)
train_predict_5 = ridge.predict(X_test)
train_predict_5
```


```python
svm = SVR(kernel='rbf')
svm.fit(X_train, y)
train_predict_6 = svm.predict(X_test)
train_predict_6
```


```python
knn = knr(n_neighbors=3)
knn.fit(X_train, y)
train_predict_7 = knn.predict(X_test)
train_predict_7
```


```python
prediction = np.expm1(train_predict_4)
output = pd.DataFrame({'Id':test_df.Id, 'SalePrice':prediction})
output.head()
```


```python
output.to_csv('xgb_all_ord.csv', index=False)
```

Our support vector regressor outperforms the others by a mile. We move on to fine tune SVR to get better performance. For practice, fine-tune the other regressors and compare how they perform.


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
        'n_estimators': randint(100, 500)
        }

random_ = RandomizedSearchCV(xgb_regg, param_distributions=params_xgb, n_iter=20, cv=5,
         scoring='neg_mean_squared_error', n_jobs=1,  random_state=42)
```


```python
random_.fit(X_train, y)
```


```python
random_.best_params_
```


```python
 random_.best_estimator_
```


```python
model = random_.best_estimator_

pred2 = model.predict(X_test)
```


```python
prediction2 = np.expm1(pred2)
output2 = pd.DataFrame({'Id':test_df.Id, 'SalePrice':prediction2})
output2.head()
```


```python
output2.to_csv('xgb8.csv', index=False)
```


```python
feature_importance = random_.best_estimator_.feature_importances_
attributes = num_attribs + ord_attribs
sorted(zip(feature_importance, attributes), reverse=True)

```


```python

```
