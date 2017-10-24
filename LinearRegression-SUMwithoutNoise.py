import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
#columns = "Feature 1 Feature 2 Feature 3 Feature 4 Feature 5 Feature 6 Feature 7 Feature 8 Feature 9 Feature 10".split()


# Load in the data with `read_csv()`
df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\The SUM dataset, without noise.csv",sep=";",nrows = 100)

X = df.loc[:,features]
y = df.Target

regressionmetric = "neg_mean_squared_error"

#X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.3)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#kf = KFold(n_splits=10)
#kf.get_n_splits(X)

#for train_index, test_index in kf.split(X):
 #   X_train, X_test = X[train_index], X[test_index]
  #  y_train, y_test = y[train_index], y[test_index]

lm = linear_model.LinearRegression()

NMSE_results= cross_val_score(lm,X,y,cv=10,scoring="neg_mean_squared_error") # Choose another regression metric

NMSE_results = NMSE_results * -1

print(NMSE_results)

NMSE_results = np.sqrt(NMSE_results)

print(NMSE_results)

#model = lm.fit(X_train,y_train)
#predictions = lm.predict(X_test)

#print(predictions[0:5])

#print('Coefficients: \n', lm.coef_)

## The line / model
#plt.scatter(y_test, predictions)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')

#print ('Score:', model.score(X_test, y_test))