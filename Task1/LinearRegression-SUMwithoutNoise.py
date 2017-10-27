import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
#columns = "Feature 1 Feature 2 Feature 3 Feature 4 Feature 5 Feature 6 Feature 7 Feature 8 Feature 9 Feature 10".split()
samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

i = 0
while i<len(samples_sizes):
    # Load in the data with `read_csv()`
    df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\The SUM dataset, without noise.csv",sep=";",nrows = samples_sizes[i])

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

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df["target_class"] = encoder.fit_transform(df["Target Class"])

    lm = linear_model.LinearRegression(normalize=True)

    NMSE_results= cross_val_score(lm,X,y,cv=10,scoring="neg_mean_squared_error") # Choose another regression metric

    NMSE_results = NMSE_results * -1

    RMS_results = np.sqrt(NMSE_results)

    mean_error = RMS_results.mean()

 #   NMSLE_results = cross_val_score(lm,X,y,cv=10,scoring="neg_mean_squared_log_error")

 #   NMSLE_results = NMSLE_results * -1

 #   MSLE_results = math.log(np.sqrt(MSLE_results))

 #   mean_error2 = mean_squared_log_error(y, lm.predict(X))

    abs_mean_error = cross_val_score(lm,X,y,cv=10,scoring="neg_mean_absolute_error")
    abs_mean_error = abs_mean_error * -1
    abs_mean_error = abs_mean_error.mean()


    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_error)

    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_error)

    i += 1
    #model = lm.fit(X_train,y_train)
    #predictions = lm.predict(X_test)

    #print(predictions[0:5])

    #print('Coefficients: \n', lm.coef_)

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel('True Values')
    #plt.ylabel('Predictions')

    #print ('Score:', model.score(X_test, y_test))