from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

i = 0
while i<len(samples_sizes):

    df = pd.read_csv("path",sep=";",nrows = samples_sizes[i])

    X = df.loc[:,features]
    y = df["Noisy Target"]

    #X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.3)

    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)

    lm = linear_model.LinearRegression(normalize=True)

    #model = lm.fit(X_train,y_train)
    #predictions = lm.predict(X_test)

    #print(predictions[0:5])

    NMSE_results= cross_val_score(lm,X,y,cv=10,scoring="neg_mean_squared_error") # Choose another regression metric

    NMSE_results = NMSE_results * -1

    RMS_results = np.sqrt(NMSE_results)

    mean_error = RMS_results.mean()

    print("Error with sample of size of ", samples_sizes[i]," = ",mean_error)


    i += 1
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