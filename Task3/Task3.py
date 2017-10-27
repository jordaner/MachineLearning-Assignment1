from sklearn import tree, linear_model, neighbors
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def normaliseScores(scores):
    old_max = max(scores)
    old_min = min(scores)
    old_range = old_max - old_min
    new_min = 0
    new_max = 1
    normalised_scores = np.array([(new_min + (((x-old_min)*(new_max-new_min)))/(old_max - old_min)) for x in scores])
    return normalised_scores

def executeAlgorithms(X, y):
    print("--- Decision Tree ---")
    model = tree.DecisionTreeRegressor()
    model = model.fit(X, y)
    getErrors(model, X, y)

    print("--- Ridge Regression ---")
    model = linear_model.Ridge(normalize = True)
    model = model.fit(X, y)
    getErrors(model, X, y)

    # print("--- SGD Regressor ---")
    #model = linear_model.SGDRegressor()
    #model = model.fit(X, y)
    #getErrors(model, X, y)

    print("--- K Nearest Neighbors ---")
    model = neighbors.KNeighborsRegressor(5)
    model = model.fit(X, y)
    getErrors(model, X, y)

    return

def getErrors(model, X, y):
    # RMS Error
    mean_squared_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_error") * -1
    root_mean_squared_error = np.sqrt(mean_squared_error)
    # Absoulte mean error
    abs_mean_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_error = abs_mean_error * -1
    abs_mean_error = abs_mean_error.mean()
    # R2 score
    r2_score = cross_val_score(model, X, y, cv=10, scoring="r2")
    # Median absolute error
    median_absolute_error = cross_val_score(model, X, y, cv=10, scoring="neg_median_absolute_error") * -1
    # Mean squared log error
    mean_squared_log_error = cross_val_score(model, X, y, cv=10, scoring="neg_mean_squared_log_error") * -1

    print("Mean squared log error =", mean_squared_log_error.mean())
    print("Median absolute error  =", median_absolute_error.mean())
    print("R2 score               =", r2_score.mean())
    print("RMS error              =", root_mean_squared_error.mean())
    print("Absolute mean error    =", abs_mean_error)

def trans_col(column):
    trans_list = LabelEncoder().fit_transform(column.tolist())
    trans_series = pd.Series(data=trans_list)
    trans_series.replace(np.NaN, 0)
    trans_series.set_value(100, 2)
    return trans_series


sampleSize = 10000
datasets = ['The SUM dataset, with noise', 'housing dataset']
SUMwithNoise_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5 (meaningless but please still use it)',
            'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
housingdataset_features = ['MSSubClass',	'MSZoning',	'LotFrontage',	'LotArea',	'Street',
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
features = [SUMwithNoise_features, housingdataset_features]
separators = [';', ',']
targets = ['Noisy Target', 'SalePrice']
# For loop iterating over the various datasets
for i in range(len(datasets)):
    print("----- " + datasets[i] + " -----")
    currentFeatures = features[i]
    currentSeparator = separators[i]
    currentTarget = targets[i]
    dataframe = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\" + datasets[i] + ".csv",
                            sep=currentSeparator,nrows=sampleSize)
    X = dataframe.loc[:, currentFeatures]
    y = dataframe[currentTarget]

    if datasets[i] == 'housing dataset':
        for column in X:
            if "object" in str(X[column].dtype): X[column] = trans_col(X[column])
        X = X.replace(np.nan, 0)

    executeAlgorithms(X, y)
