from src.models.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
my_reg = LinearRegression()
my_reg.fit(X, y, iters=10000)
reg = sklearnLinearRegression()
reg.fit(X, y)
print(my_reg.w)
print(my_reg.b)
print(reg.coef_)
print(reg.intercept_)