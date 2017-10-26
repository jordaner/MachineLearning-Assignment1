NMSE_resultsWithoutRR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_resultsWithoutRR = NMSE_resultsWithoutRR * -1
RMS_resultsWithoutRR = np.sqrt(NMSE_resultsWithoutRR)  # LINEAR REGRESSION SUM WITHOUT NOISE
mean_errorWithoutRR = RMS_resultsWithoutRR.mean()
abs_mean_errorWithoutRR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
abs_mean_errorWithoutRR = abs_mean_errorWithoutRR * -1
abs_mean_errorWithoutRR = abs_mean_errorWithoutRR.mean()