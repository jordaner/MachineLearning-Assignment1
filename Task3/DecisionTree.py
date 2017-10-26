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

samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithNoise",sep=";",nrows = samples_sizes[0])

X = df.loc[:,features]
y = df["Noisy Target"]

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
NMSE_results= cross_val_score(clf,X,y,cv=10,scoring="neg_mean_squared_error")

mean_error = getrmse(NMSE_results)

print(mean_error)