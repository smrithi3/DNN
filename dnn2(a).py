# 2(a) Design a single unit perceptron for classification of a linearly separable binary dataset without using pre-defined models.Use the perceptron() from sklearn .
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Generate a linearly separable binary dataset
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 1, 1, 0)

# Train a Perceptron model
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=0)
perceptron.fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the decision boundary and data points
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')
plt.show()