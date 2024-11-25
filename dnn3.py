# 3.Build a deep feed forward ANN by implementing the backpropagation algorithm and test the same using appropriate datasets . Use the number of hidden layers >=4.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Activation Functions and their derivatives
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return Z * (1 - Z)

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

# Neural Network class
class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        self.params = self.initialize_weights()

    def initialize_weights(self):
        params = {}
        for i in range(1, len(self.layers)):
            params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01
            params['b' + str(i)] = np.zeros((self.layers[i], 1))
        return params

    def forward_propagation(self, X):
        cache = {'A0': X}
        A = X
        for i in range(1, len(self.layers)):
            Z = np.dot(self.params['W' + str(i)], A) + self.params['b' + str(i)]
            A = relu(Z) if i < len(self.layers) - 1 else sigmoid(Z)
            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A
        return A, cache

    def backward_propagation(self, Y, cache):
        grads = {}
        m = Y.shape[1]
        A_final = cache['A' + str(len(self.layers) - 1)]
        dZ = A_final - Y  # Sigmoid output layer

        for i in reversed(range(1, len(self.layers))):
            dW = (1/m) * np.dot(dZ, cache['A' + str(i - 1)].T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            grads['dW' + str(i)] = dW
            grads['db' + str(i)] = db

            if i > 1:
                dZ_prev = np.dot(self.params['W' + str(i)].T, dZ)
                dZ = dZ_prev * relu_derivative(cache['Z' + str(i - 1)])

        return grads

    def update_parameters(self, grads):
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] -= self.learning_rate * grads['dW' + str(i)]
            self.params['b' + str(i)] -= self.learning_rate * grads['db' + str(i)]

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            A, cache = self.forward_propagation(X)
            cost = self.compute_cost(A, Y)
            grads = self.backward_propagation(Y, cache)
            self.update_parameters(grads)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, cost: {cost:.4f}')

    def compute_cost(self, A_final, Y):
        m = Y.shape[1]
        return (-1/m) * np.sum(Y * np.log(A_final) + (1 - Y) * np.log(1 - A_final))

    def predict(self, X):
        A, _ = self.forward_propagation(X)
        return np.argmax(A, axis=0)

# Load dataset and prepare data
iris = load_iris()
X = iris.data.T
y = iris.target.reshape(-1, 1)

# One-hot encoding of labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.T, y_onehot.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T

# Set up and train neural network
input_size = X_train.shape[0]
output_size = y_train.shape[0]
hidden_layers = [10, 8, 8, 6]  # Number of neurons in hidden layers
learning_rate = 0.01

nn = DeepNeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
nn.train(X_train, y_train, epochs=1000)

# Test the network
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

# Convert one-hot predictions back to single class
y_train_labels = np.argmax(y_train, axis=0)
y_test_labels = np.argmax(y_test, axis=0)

# Calculate accuracy
train_accuracy = accuracy_score(y_train_labels, y_pred_train)
test_accuracy = accuracy_score(y_test_labels, y_pred_test)

print(f'Training accuracy: {train_accuracy * 100:.2f}%')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
