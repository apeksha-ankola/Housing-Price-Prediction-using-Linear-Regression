# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset and preprocess
data = pd.read_csv('Housing Price data set.csv')


# Remove unnecessary data
data = data.drop(['Unnamed: 0', 'driveway', 'recroom','fullbase', 'airco', 'prefarea', 'gashw'], axis=1)
mean = data.mean()[0]
stddev = data.std()[0]
data = (data - data.mean()) / data.std()


# Separate features and target
data = np.asarray(data)
Y = data[:, 0:1]
X = data[:, 1:]
one = np.ones((len(X), 1))
X = np.concatenate((one, X), axis=1)


# Split data into train and test sets
split_ratio = 0.9
split = int(split_ratio * X.shape[0])
X_test = X[split+1:, :]
X_train = X[:split+1, :]
Y_test = Y[split+1:, :]
Y_train = Y[:split+1, :]

# Function to denormalize price
def denormalise_price(price):
  global mean, stddev
  ret = price * stddev + mean
  return ret


# Normal Equation with regularization
def normalEquation(X, Y, lam):
  lam_matrix = lam * np.identity(X.shape[1])
  lam_matrix[0][0] = 0 # No regularization on bias term
  theta = np.linalg.inv(X.T.dot(X) + 
lam_matrix).dot(X.T).dot(Y)
  return theta


# Train using Normal Equation
lam = 0 
# Regularization parameter (set to 0 for no regularization)
theta = normalEquation(X_train, Y_train, lam)
# Predict using test data
Y_pred = X_test.dot(theta)


# Compute error
def computeError(predicted, actual):
  error = np.mean(np.abs((actual - predicted) / actual)) * 100
  return error
error = computeError(denormalise_price(Y_pred), 
denormalise_price(Y_test))

# Example prediction
print("Mean Absolute Percentage Error:", error, "%")
print('Predicted price =', denormalise_price(Y_pred[2][0]))
print('Actual price =', denormalise_price(Y_test[2][0]))

# mean square error calculations
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
