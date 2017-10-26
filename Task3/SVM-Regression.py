# Source in api http://scikit-learn.org/stable/modules/svm.html#regression
# datasets are: SUMwithNoise and HousePrices

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
from sklearn import svm

features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/winequality-white.csv", sep=";")

X = df.loc[:,features]
Y = df.quality
kf = KFold(n_splits=10)


def evaluate(X, Y, kf):
    linReg = LinearRegression()
    svmReg = svm.SVR()
    decTreeReg = tree.DecisionTreeRegressor()
    sgdReg = SGDRegressor()
    algos = np.array([linReg, svmReg, decTreeReg, sgdReg])
    for i in range(0, algos.size):
        print("ALGORITHM:", str(algos[i]))
        abs_error = cross_val_score(algos[i], X, Y, cv = kf, scoring ='neg_mean_absolute_error')
        mean_score = abs_error.mean()
        print("mean absolute error:", -1 * mean_score)

        sq_error = cross_val_score(algos[i], X, Y, cv=kf, scoring='neg_mean_squared_error')
        mean_sqerror = -1 * sq_error.mean()
        print("mean squared error: ", np.sqrt(mean_sqerror))

        med_abs_error = cross_val_score(algos[i], X, Y, cv=kf, scoring='neg_median_absolute_error')
        med_abs_error = -1 * med_abs_error
        print("median absolute error: ", med_abs_error)

        r2 = cross_val_score(algos[i], X, Y, cv=kf, scoring='r2')
        print("r2: ",r2)

        expl_var = cross_val_score(algos[i], X, Y, cv=kf, scoring='explained_variance')
        print("explained var: ",expl_var)


svm_regressor = svm.SVR()
abs_error = cross_val_score(svm_regressor, X, Y, cv = kf, scoring ='neg_mean_absolute_error')
mean_score = abs_error.mean()
print("mean absolute error:", -1 * mean_score)



