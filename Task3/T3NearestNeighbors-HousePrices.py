# Source in api http://scikit-learn.org/stable/modules/tree.html#regression
# datasets are: SUMwithNoise and HousePrices

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn import neighbors, datasets
from sklearn import neighbors


def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series.replace(np.NaN, 0)
    transformed_series.set_value(100, 2)
    return transformed_series

def transformValueToClassValue(value):
    if "str" in str(type(value)):
        return value
    else:
        return round(value/100000)

def getrmse(number):
    number = number * -1

    number = np.sqrt(number)

    number = number.mean()
    return number

features = ['MSSubClass',	'MSZoning',	'LotFrontage',	'LotArea',	'Street',
            'Alley',	'LotShape',	'LandContour',	'Utilities',	'LotConfig',
            'LandSlope',	'Neighborhood',	'Condition1',	'Condition2',	'BldgType',
            'HouseStyle',	'OverallQual',	'OverallCond',
            'YearBuilt',	'YearRemodAdd',	'RoofStyle',	'RoofMatl',
            'Exterior1st',	'Exterior2nd',	'MasVnrType',	'MasVnrArea',	'ExterQual',
            'ExterCond',	'Foundation',	'BsmtQual',	'BsmtCond',	'BsmtExposure',
            'BsmtFinType1',	'BsmtFinSF1',	'BsmtFinType2',	'BsmtFinSF2',
            'BsmtUnfSF',	'TotalBsmtSF',	'Heating',	'HeatingQC',	'CentralAir',
            'Electrical',	'1stFlrSF',	'2ndFlrSF',	'LowQualFinSF',	'GrLivArea',	'BsmtFullBath',
            'BsmtHalfBath',	'FullBath',	'HalfBath',	'BedroomAbvGr',	'KitchenAbvGr',
            'KitchenQual',	'TotRmsAbvGrd',	'Functional',	'Fireplaces',	'FireplaceQu',	'GarageType',
            'GarageYrBlt',	'GarageFinish'	'GarageCars',	'GarageArea',
            'GarageQual',	'GarageCond',	'PavedDrive',	'WoodDeckSF',	'OpenPorchSF',
            'EnclosedPorch',	'3SsnPorch',	'ScreenPorch',	'PoolArea',	'PoolQC',	'Fence',
            'MiscFeature',	'MiscVal',	'MoSold',	'YrSold',	'SaleType',	'SaleCondition']

df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/housing dataset.csv",sep=",",nrows = 10000)

X = df.loc[:,features]
y = df.SalePrice

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    clf = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
for column in X:
    if "object" in str(X[column].dtype):
        X[column] = transformColumn(X[column])
X = X.replace(np.nan, 0)

y = [transformValueToClassValue(i) for i in (y.tolist())]
y = pd.Series(data=y)

clf = clf.fit(X, y)
NMSE_results= cross_val_score(clf,X,y,cv=10,scoring="neg_mean_squared_error")

mean_error = getrmse(NMSE_results)

print("Error with sample size 10000 is",mean_error)
