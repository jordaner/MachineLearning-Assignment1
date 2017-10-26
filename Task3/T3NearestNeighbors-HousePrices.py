# Source in api http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression
# dataset is: HousePrices

# Source in api http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression
# datasets are: SUMwithNoise and HousePrices
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold
#from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import neighbors, datasets

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



df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\The SUM dataset, with noise.csv",sep=";",nrows = samples_sizes[i])

catnum = df["Noisy Target Class"].tolist()

X = df.loc[:,features]
y_a = le.fit(catnum)
y_b = le.transform(catnum)

lr = neighbors.KNeighborsClassifier(len(samples_sizes))

#Needed to avoid members of target class being less that the number of folds
kfold = KFold(n_splits=10,random_state=0)
ACC_results = cross_val_score(lr, X, y_b, cv=kfold, scoring="accuracy")

PREC_scorer =  make_scorer(precision_score, average="weighted")
PREC_results = cross_val_score(lr, X, y_b, cv=kfold, scoring=PREC_scorer)

mean_ACC = ACC_results.mean()
mean_PREC = PREC_results.mean()

print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACC)
print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PREC)

i += 1