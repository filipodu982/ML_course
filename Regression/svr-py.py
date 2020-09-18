# Importing libraries
import numpy as np
import pandas as pd
# Importing necessary classes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Reading dataset and creating independent and dependent variable matrices
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Reshaping for feature scaling
y = y.reshape(len(y), 1)

# Splitting data for training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Creating Linear Regressor  and training it
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

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