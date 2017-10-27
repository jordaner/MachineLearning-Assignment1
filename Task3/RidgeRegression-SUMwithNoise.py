import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
samples_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
i = 0

while i < len(samples_sizes):
    data = pd.read_csv("C://Users//Bernard/Dropbox//TCD//CS4404//Machine Learning Datasets//The SUM dataset//with noise//The SUM dataset, with noise.csv", sep=";" , nrows = samples_sizes[i])
    X, y = data.loc[:,features], data["Noisy Target"]
    reg_metric = "neg_mean_squared_error"
    lin_mod = linear_model.Ridge(normalize=True)
    NMSE_results= cross_val_score(lin_mod, X, y, cv=10, scoring=reg_metric) # Choose another regression metric
    NMSE_results = NMSE_results * -1
    RMS_results = np.sqrt(NMSE_results)
    mean_error = RMS_results.mean()
    abs_mean_error = cross_val_score(lin_mod, X, y, cv=10, scoring=reg_metric)
    abs_mean_error = abs_mean_error * -1
    abs_mean_error = abs_mean_error.mean()
    print("Error with sample size of ", samples_sizes[i], "for mean squared error = ", mean_error)
    print("Error with sample size of ", samples_sizes[i], "for absolute mean error =", abs_mean_error)
    i += 1