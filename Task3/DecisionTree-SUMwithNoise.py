# Source in api http://scikit-learn.org/stable/modules/tree.html#regression
# datasets are: SUMwithNoise and HousePrices

from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


def getrmse(number):
    number = number * -1

    number = np.sqrt(number)

    number = number.mean()
    return number

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']

df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\The SUM dataset, with noise.csv",sep=";")

X = df.loc[:,features]
y = df["Noisy Target"]

model = tree.DecisionTreeRegressor()
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
