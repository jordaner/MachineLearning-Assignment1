# Source in api http://scikit-learn.org/stable/modules/tree.html#regression
# datasets are: SUMwithNoise and HousePrices

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

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

df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\housing dataset.csv",sep=",")
X = df.loc[:,features]
y = df.SalePrice

model = tree.DecisionTreeRegressor()
for column in X:
    if "object" in str(X[column].dtype):
        X[column] = transformColumn(X[column])
X = X.replace(np.nan, 0)

y = [transformValueToClassValue(i) for i in (y.tolist())]
y = pd.Series(data=y)

model = model.fit(X, y)

# RMS Error
mean_squared_error= cross_val_score(model,X,y,cv=10,scoring="neg_mean_squared_error") * -1
root_mean_squared_error = np.sqrt(mean_squared_error)
# Absoulte mean error
abs_mean_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")
abs_mean_error = abs_mean_error * -1
abs_mean_error = abs_mean_error.mean()
# R2 score
r2_score = cross_val_score(model, X, y, cv = 10, scoring = "r2")
# Median absolute error
median_absolute_error = cross_val_score(model, X, y, cv = 10, scoring = "neg_median_absolute_error") * -1
# Mean squared log error
mean_squared_log_error = cross_val_score(model, X, y, cv = 10, scoring = "neg_mean_squared_log_error") * -1

print("Mean squared log error with sample size of 10000 =", mean_squared_log_error.mean())
print("Median absolute error with sample size of 10000 =", median_absolute_error.mean())
print("R2 score with sample size of 10000 =", r2_score.mean())
print("RMS error with sample size of 10000 =", root_mean_squared_error)
print("Absolute mean error with sample size of 10000 for absolute mean error =", abs_mean_error)