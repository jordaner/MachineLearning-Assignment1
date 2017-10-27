from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def normaliseScores(scores):
    old_max = max(scores)
    old_min = min(scores)
    old_range = old_max - old_min
    new_min = 0
    new_max = 1
    normalised_scores = np.array([(new_min + (((x-old_min)*(new_max-new_min)))/(old_max - old_min)) for x in scores])
    return normalised_scores


features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
samples_sizes= [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

i = 0
while i<len(samples_sizes):

    df = pd.read_csv("/Users/markloughman/Desktop/Machine Learning/DATA/TheSumDataSetWithNoise",sep=";",nrows = samples_sizes[i])

    X = df.loc[:,features]
    y = df["Noisy Target"]

    lm = linear_model.LinearRegression(normalize=True)

    NMSE_results = cross_val_score(lm, X, y, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
    NMSE_results = NMSE_results * -1
    RMS_results = np.sqrt(NMSE_results)
    RMS_results = normaliseScores(RMS_results)
    mean_error = RMS_results.mean()

    abs_mean_error = cross_val_score(lm, X, y, cv=10, scoring="neg_mean_absolute_error")
    abs_mean_error = abs_mean_error * -1
    abs_mean_error = normaliseScores(abs_mean_error)
    abs_mean_error = abs_mean_error.mean()

    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_error)
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_error)

    i += 1