import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series = pd.Series(data=transformed_list)
    transformed_series.replace(np.NaN, 0)
    transformed_series.set_value(100, 2)
    return transformed_series

sumFeatures = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

whiteWineFeatures = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

housePriceFeatures = ['MSSubClass',	'MSZoning',	'LotFrontage',	'LotArea',	'Street',
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

i = 0
while i<len(samples_sizes):

    sumWithoutNoise = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithoutNoise", sep=";", nrows=samples_sizes[i])

    sumWithNoise = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithNoise", sep=";",nrows=samples_sizes[i])

    whiteWine = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/winequality-white.csv", sep=";", nrows=samples_sizes[i])

    housePrices = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/housing dataset.csv",sep=",",nrows = samples_sizes[i])

    sumWithoutX = sumWithoutNoise.loc[:, sumFeatures]
    sumWithoutY = sumWithoutNoise.Target

    sumWithX = sumWithNoise.loc[:, sumFeatures]
    sumWithY = sumWithNoise["Noisy Target"]

    wineX = whiteWine.loc[:, whiteWineFeatures]
    wineY = whiteWine.quality

    houseX = housePrices.loc[:, housePriceFeatures]
    houseY = housePrices.SalePrice

    lm = linear_model.LinearRegression(normalize=True)
    rr = linear_model.Ridge(normalize=True)

    NMSE_resultsWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithoutLR = NMSE_resultsWithoutLR * -1
    RMS_resultsWithoutLR = np.sqrt(NMSE_resultsWithoutLR)                                                                           #LINEAR REGRESSION SUM WITHOUT NOISE
    mean_errorWithoutLR = RMS_resultsWithoutLR.mean()
    abs_mean_errorWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithoutLR = abs_mean_errorWithoutLR * -1
    abs_mean_errorWithoutLR = abs_mean_errorWithoutLR.mean()

    NMSE_resultsWithoutRR = cross_val_score(rr, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithoutRR = NMSE_resultsWithoutRR * -1
    RMS_resultsWithoutRR = np.sqrt(NMSE_resultsWithoutRR)  # LINEAR REGRESSION SUM WITHOUT NOISE
    mean_errorWithoutRR = RMS_resultsWithoutRR.mean()                                                                                #RIDGE REGRESSION SUM WITHOUT NOISE
    abs_mean_errorWithoutRR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithoutRR = abs_mean_errorWithoutRR * -1
    abs_mean_errorWithoutRR = abs_mean_errorWithoutRR.mean()

    NMSE_resultsWithout = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWithout = NMSE_resultsWithout * -1
    RMS_resultsWithout = np.sqrt(NMSE_resultsWithout)                                                                           #LINEAR REGRESSION SUM WITHOUT NOISE
    mean_errorWithout = RMS_resultsWithout.mean()
    abs_mean_errorWithout = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWithout = abs_mean_errorWithout * -1
    abs_mean_errorWithout = abs_mean_errorWithout.mean()



    NMSE_resultsWith = cross_val_score(lm, sumWithX, sumWithY, cv=10, scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsWith = NMSE_resultsWith * -1
    RMS_resultsWith = np.sqrt(NMSE_resultsWith)                                                                                  #LINEAR REGRESSION SUM WITH NOISE
    mean_errorWith = RMS_resultsWith.mean()
    abs_mean_errorWith = cross_val_score(lm, sumWithX, sumWithY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWith = abs_mean_errorWith * -1
    abs_mean_errorWith = abs_mean_errorWith.mean()

    NMSE_resultsWhiteWine= cross_val_score(lm,wineX,wineY,cv=10,scoring="neg_mean_squared_error") # Choose another regression metric
    NMSE_resultsWhiteWine = NMSE_resultsWhiteWine * -1
    RMS_resultsWhiteWine = np.sqrt(NMSE_resultsWhiteWine)                                                           #LINEAR REGRESSION WHITE WINE DATA SET
    mean_errorWhiteWine = RMS_resultsWhiteWine.mean()
    abs_mean_errorWhiteWine = cross_val_score(lm, wineX, wineY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorWhiteWine = abs_mean_errorWhiteWine * -1
    abs_mean_errorWhiteWine = abs_mean_errorWhiteWine.mean()

    encoder = LabelEncoder()
    housePrices["target_class"] = encoder.fit_transform(housePrices["SalePrice"].tolist())

    for column in houseX:
        if "object" in str(houseX[column].dtype):
            houseX[column] = transformColumn(houseX[column])
    houseX = houseX.replace(np.nan,0)

    NMSE_resultsHouse = cross_val_score(lm, houseX, houseY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_resultsHouse = NMSE_resultsHouse * -1
    RMS_resultsHouse = np.sqrt(NMSE_resultsHouse)
    mean_errorHouse = RMS_resultsHouse.mean()                                                                         #LINEAR REGRESSION HOUSE PRICES DATA SET
    abs_mean_errorHouse = cross_val_score(lm, houseX, houseY, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_errorHouse = abs_mean_errorHouse * -1
    abs_mean_errorHouse = abs_mean_errorHouse.mean()

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorHouse, "for the house prices data set")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorHouse, "for the house prices data set")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWhiteWine, "for the White Wine Data Set")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWhiteWine, "for the White Wine Data Set")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWith, "for SUM with Noise")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWith, "for SUM with Noise")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithoutLR, "for SUM without Noise - Linear Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithoutLR, "for SUM without Noise - Linear Regression")

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_errorWithoutRR, "for SUM without Noise - Ridge Regression")
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_errorWithoutRR,"for SUM without Noise - Ridge Regression")

    i += 1
