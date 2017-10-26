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
from sklearn import neighbors

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
le = preprocessing.LabelEncoder()

#while i<len(samples_sizes):

df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithNoise",sep=";",nrows = 10000)

catnum = df["Noisy Target Class"].tolist()

X = df.loc[:,features]
y = df["Noisy Target"]

n_neighbors = 5


for i, weights in enumerate(['uniform', 'distance']):
    lr = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

#Needed to avoid members of target class being less that the number of folds
NMSE_results = cross_val_score(lr, X, y, cv=10,
                                   scoring="neg_mean_squared_error")  # Choose another regression metric

NMSE_results = NMSE_results * -1

RMS_results = np.sqrt(NMSE_results)

mean_error = RMS_results.mean()

abs_mean_error = cross_val_score(lr, X, y, cv=10, scoring="neg_mean_absolute_error")
abs_mean_error = abs_mean_error * -1
abs_mean_error = abs_mean_error.mean()

print("Error with sample size of 10000 for mean squared error = ", mean_error)

print("Error with sample size of 10000 for absolute mean error =", abs_mean_error)


 #   i += 1