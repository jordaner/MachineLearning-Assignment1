# Source in api http://scikit-learn.org/stable/modules/svm.html#regression
# datasets are: SUMwithNoise and HousePrices

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\The SUM dataset, with noise.csv", sep=";")

X = df.loc[:, features]
Y = df["Noisy Target"]
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


sgd_regressor = SGDRegressor()
abs_error = cross_val_score(sgd_regressor, X, Y, cv = kf, scoring ='neg_mean_absolute_error')
mean_score = abs_error.mean()
print("mean absolute error:", -1 * mean_score)




