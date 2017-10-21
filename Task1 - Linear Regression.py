import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5 (meaningless but please still use it)',
            'Feature 6','Feature 7','Feature 8','Feature 9','Feature 10','Target','Target Class']

# Load in the data with `read_csv()`
digits = pd.read_csv("C:\Users\ericj\PycharmProjects\Assignment1\MachineLearning-Assignment1\The SUM dataset, "
                     "without noise.csv",sep=";",nrows=100)

# Extract the features
X = digits.loc[:,features]

#pd.get_dummies(digits)
# Print out `digits`
print(digits)
print(X)

# Split the data into training/testing sets
digits_train = X[:-20]
digits_test = X[-20:]

# Split the targets into training/testing sets
digits_y_train = digits.Target[:-20]
digits_y_test = digits.Target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(digits_train, digits_y_train)

# Make predictions using the testing set
digits_y_pred = regr.predict(digits_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(digits_y_test, digits_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(digits_y_test, digits_y_pred))

# Plot outputs
plt.scatter(digits_test, digits_y_test,  color='black')
plt.plot(digits_test, digits_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()