from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold
#from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score, make_scorer

#wine and housing data set tbh

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
le = preprocessing.LabelEncoder()

i = 0
while i<len(samples_sizes):

    df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithNoise",sep=";",nrows = samples_sizes[i])

    catnum = df["Noisy Target Class"].tolist()

    X = df.loc[:,features]
    y_a = le.fit(catnum)
    y_b = le.transform(catnum)

    lr = linear_model.LogisticRegression()

    #Needed to avoid members of target class being less that the number of folds
    kfold = KFold(n_splits=10,random_state=0)
    ACC_results = cross_val_score(lr, X, y_b, cv=kfold, scoring="accuracy")

    PREC_scorer =  make_scorer(precision_score, average="weighted")
    PREC_results = cross_val_score(lr, X, y_b, cv=kfold, scoring=PREC_scorer)

    mean_ACC = ACC_results.mean()
    mean_PREC = PREC_results.mean()

    print("Accuracy with sample of size of ", samples_sizes[i], " = ", mean_ACC)
    print("Precision Score with sample of size of ", samples_sizes[i], " = ", mean_PREC)


    i += 1