from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
le = preprocessing.LabelEncoder()

i = 0
while i<len(samples_sizes):

    df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/housing dataset.csv",sep=",",nrows = samples_sizes[i])
    X = df.loc[:,features]
    y = df.SalePrice

    lm = linear_model.LogisticRegression()

    for column in X:
        if "object" in str(X[column].dtype):
            X[column] = transformColumn(X[column])
    X = X.replace(np.nan,0)


    y = [transformValueToClassValue(i) for i in (y.tolist())]
    y = pd.Series(data=y)




    kfold = KFold(n_splits=10, random_state=0)
    ACC_results = cross_val_score(lm, X, y, cv=kfold, scoring="accuracy")

    PREC_scorer = make_scorer(precision_score, average="weighted")
    PREC_results = cross_val_score(lm, X, y, cv=kfold, scoring=PREC_scorer)

    mean_ACC = ACC_results.mean()
    mean_PREC = PREC_results.mean()

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACC)
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PREC)

    i += 1