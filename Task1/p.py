NMSE_resultsWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10,
                                        scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_resultsWithoutLR = NMSE_resultsWithoutLR * -1
RMS_resultsWithoutLR = np.sqrt(NMSE_resultsWithoutLR)  # LINEAR REGRESSION SUM WITHOUT NOISE
RMS_resultsWithoutLR = normaliseScores(RMS_resultsWithoutLR)
mean_errorWithoutLR = RMS_resultsWithoutLR.mean()
abs_mean_errorWithoutLR = cross_val_score(lm, sumWithoutX, sumWithoutY, cv=10, scoring="neg_mean_absolute_error")
abs_mean_errorWithoutLR = abs_mean_errorWithoutLR * -1
abs_mean_errorWithoutLR = normaliseScores(abs_mean_error)
abs_mean_errorWithoutLR = abs_mean_errorWithoutLR.mean()

NMSE_results = cross_val_score(lm, X, y, cv=10, scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_results = NMSE_results * -1
RMS_results = np.sqrt(NMSE_results)
RMS_results = normaliseScores(RMS_results)
mean_error = RMS_results.mean()
abs_mean_error = cross_val_score(lm, X, y, cv=10, scoring="neg_mean_absolute_error")
abs_mean_error = abs_mean_error * -1
abs_mean_error = normaliseScores(abs_mean_error)
abs_mean_error = abs_mean_error.mean()

NMSE_resultsHouseRR = cross_val_score(rr, houseX, houseY, cv=10,scoring="neg_mean_squared_error")  # Choose another regression metric
NMSE_resultsHouseRR = NMSE_resultsHouseRR * -1
RMS_resultsHouseRR = np.sqrt(NMSE_resultsHouseRR)
RMS_resultsHouseRR = normaliseScores(RMS_resultsHouseRR)
mean_errorHouseRR = RMS_resultsHouseRR.mean()  # RIDGE REGRESSION HOUSE PRICES DATA SET
abs_mean_errorHouseRR = cross_val_score(rr, houseX, houseY, cv=10, scoring="neg_mean_absolute_error")
abs_mean_errorHouseRR = abs_mean_errorHouseRR * -1
abs_mean_errorHouseRR = normaliseScores(abs_mean_errorHouseRR)
abs_mean_errorHouseRR = abs_mean_errorHouseRR.mean()

