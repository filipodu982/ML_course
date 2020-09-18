# Importing libraries
import numpy as np
import pandas as pd
# Importing necessary classes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Reading dataset and creating independent and dependent variable matrices
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data for training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating polynomial features
pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X_train)

# Creating Linear Regressor  and training it
lr = LinearRegression()
lr.fit(X_poly, y_train)

# Predicting test set results
y_pred = lr.predict(pr.transform(X_test))

# Printing options
np.set_printoptions(precision=2)

# Reshaping vectors so they are vertical and not horizontal
y_pred = y_pred.reshape(len(y_pred), 1)
y_test = y_test.reshape(len(y_test), 1)

# For printing test and predicted results side by side
results = np.concatenate((y_pred, y_test), 1)


print(results)

# Printing % of error
print(100 * abs(results[:, 0] - results[:, 1]) / results[:, 1])
print(r2_score(y_test, y_pred))