import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
i = 500
while i<=500000000:
    # Load in the data with `read_csv()`
    df = pd.read_csv("path",sep=";",nrows = i)

    X = df.loc[:,features]
    y = df.Target

    X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.3)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    lm = linear_model.LinearRegression()

    model = lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)

    #print(predictions[0:5])

    # The coefficients
    print('Without Noise Coefficients: \n', lm.coef_)
    #    The mean squared error
    print("Without Noise Mean squared error: %.2f"
          % mean_squared_error(y_test, predictions))
    # Explained variance score: 1 is perfect prediction
    print('Without Noise Variance score: %.2f' % r2_score(y_test, predictions))

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel('True Values')
    #plt.ylabel('Predictions')

    print ('Score:', model.score(X_test, y_test))

    i = i*10
