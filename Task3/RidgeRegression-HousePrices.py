import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def trans_col(column):
    trans_list = LabelEncoder().fit_transform(column.tolist())
    trans_series = pd.Series(data=trans_list)
    trans_series.replace(np.NaN, 0)
    trans_series.set_value(100, 2)
    return trans_series

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

data = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\housing dataset.csv",sep=",")
X = data.loc[:,features]
y = data.SalePrice
lin_mod = linear_model.Ridge(normalize = True)
for column in X:
    if "object" in str(X[column].dtype): X[column] = trans_col(X[column])
X = X.replace(np.nan,0)
NMSE_results= cross_val_score(lin_mod, X, y, cv = 10, scoring = "neg_mean_squared_error") # Choose another regression metric
NMSE_results = NMSE_results * -1
RMS_results = np.sqrt(NMSE_results)
mean_error = RMS_results.mean()
abs_mean_error = cross_val_score(lin_mod, X, y, cv = 10, scoring = "neg_mean_absolute_error")
abs_mean_error = abs_mean_error * -1
abs_mean_error = abs_mean_error.mean()
print("Mean squared error = ", mean_error)
print("Absolute mean error =", abs_mean_error)
