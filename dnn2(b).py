# Identify the problem with single unit perceptron . Classify using OR , AND , XOR data and analyse the result 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Function to train perceptron on given data
def train_perceptron(X, y):
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=0)
    perceptron.fit(X, y)
    return perceptron

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# OR Dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
model_or = train_perceptron(X_or, y_or)
plot_decision_boundary(X_or, y_or, model_or, 'Perceptron Decision Boundary for OR')

# AND Dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
model_and = train_perceptron(X_and, y_and)
plot_decision_boundary(X_and, y_and, model_and, 'Perceptron Decision Boundary for AND')

# XOR Dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
model_xor = train_perceptron(X_xor, y_xor)
plot_decision_boundary(X_xor, y_xor, model_xor, 'Perceptron Decision Boundary for XOR')