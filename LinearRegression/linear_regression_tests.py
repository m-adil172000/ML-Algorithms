import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


# Testing Simple Linear Regression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# plt.scatter(X_train, y_train)
# plt.show()


# Using the sklearn library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)
print(predicted)
def mse(y_true, y_predicted):
    return np.sum((y_true - y_predicted)**2) / len(y_true)

mse_value = mse(y_test, predicted)  
print(f"mse_scikit-learn: {mse_value}")

'''
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m, b = np.polyfit(X, y, 1)
plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
plt.plot(X, y_pred_line, color=cmap(0.5), linewidth=2, label='Scikit-learn Prediction')
plt.show()
print(f"coefficient: {m}, intercept: {b}")
'''

# Using our own class
from Simple_linear_regression import simpleLR


reg = simpleLR()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print(pred)

mse_val = mse(y_test, pred)  
print(f"mse_simpleLR: {mse_value}")

'''
y_pred_line_2 = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m_2, b_2 = np.polyfit(X, y, 1)
plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
plt.plot(X, y_pred_line_2, color=cmap(0.5), linewidth=2, label='simpleLR Prediction')
plt.show()
print(f"coefficient: {m_2}, intercept: {b_2}")
'''