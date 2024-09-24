from src.models.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
import numpy as np
import time

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

start_time = time.time()
my_reg = LinearRegression()
my_reg.fit(X, y, iters=10000, info=False)
my_time = time.time() - start_time

start_time = time.time()
sklearn_reg = sklearnLinearRegression()
sklearn_reg.fit(X, y)
sklearn_time = time.time() - start_time

print("My implementation:")
print(f"Time: {my_time} seconds")
print(f"Weights: {my_reg.w}")
print(f"Bias: {my_reg.b}")

print("\nsklearn implementation:")
print(f"Time: {sklearn_time} seconds")
print(f"Weights: {sklearn_reg.coef_}")
print(f"Bias: {sklearn_reg.intercept_}")

print(f"Time difference: {abs(my_time - sklearn_time):.6f}")
if my_time < sklearn_time:
    print(f"My implementation was {sklearn_time / my_time:.2f} times faster")
else:
    print(f"sklearn's implementation was {my_time / sklearn_time:.2f} times faster")