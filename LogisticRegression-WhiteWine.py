from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']
samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

i = 0
while i<len(samples_sizes):

    df = pd.read_csv("C:\\Users\\ericj\\PycharmProjects\\Assignment1\\winequality-white.csv",sep=";",nrows = samples_sizes[0])

    X = df.loc[:,features]
    y = df["quality"]

    print(X)
    print(y)
    #X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.3)

    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)


  #  from sklearn.preprocessing import LabelEncoder
   # encoder = LabelEncoder()
    #df["target_class"] = encoder.fit_transform(df["Target Class"])

    #lm = linear_model.LinearRegression(normalize=True)

    #model = lm.fit(X_train,y_train)
    #predictions = lm.predict(X_test)

    #print(predictions[0:5])

    #NMSE_results= cross_val_score(lm,X,y,cv=10,scoring="neg_mean_squared_error") # Choose another regression metric

    #NMSE_results = NMSE_results * -1

    #RMS_results = np.sqrt(NMSE_results)

    #mean_error = RMS_results.mean()


    #abs_mean_error = cross_val_score(lm,X,y,cv=10,scoring="neg_mean_absolute_error")
    #abs_mean_error = abs_mean_error * -1
    #abs_mean_error = abs_mean_error.mean()


    #print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_error)

    #print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_error)




    #i += 1
    # The coefficients
    #print('With Noise Coefficients: \n', lm.coef_)
    #    The mean squared error
    #print("With Noise Mean squared error: %.2f"
     #     % mean_squared_error(y_test, predictions))
    # Explained variance score: 1 is perfect prediction
    #print('With Noise Variance score: %.2f' % r2_score(y_test, predictions))

    ## The line / model
    #plt.scatter(y_test, predictions)
    #plt.xlabel('True Values')
    #plt.ylabel('Predictions')

    #print ('Score:', model.score(X_test, y_test))