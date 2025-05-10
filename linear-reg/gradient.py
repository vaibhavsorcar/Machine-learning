import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x,y = make_regression(n_samples = 100, n_features =1, noise= 15, random_state = 42)
y = y.reshape(-1, 1)
m = x.shape[0]
# print(m)

xb = np.c_[np.ones((m, 1)), x]
theta = np.array([[2.0], [3.0]])
plt.figure(figsize = (10, 5))
plt.scatter(x, y, color = "blue",  label = "Acutal data")
plt.plot(x, xb.dot(theta), color = "red", label = "Initial Line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear regression wihtout gradient descent")
plt.legend()
plt.show()

# -------------------------------------------------------
"""
learning_rate = 0.1
n_iterations = 100
for i in range(n_iterations):
    y_pred = xb.dot(theta)
    gradients = (2/m)*xb.T.dot(y_pred - y)
    theta -=learning * gradients

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, X_b.dot(theta), color="red", label="Optimized Line (With GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()
"""
# -------------------------------------------------------