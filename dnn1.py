# 1. Write a program to demonstrate the working of different activation function like Sigmoid , Tanh , RELU and softmax to train neural network .
import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
   
    return np.tanh(x)

def relu(x):
    
    return np.maximum(x, 0)

def softmax(x):
    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Generate input data
x = np.linspace(-10, 10, 100)

# Apply activation functions
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)
relu_output = relu(x)

# Softmax requires a 2D input, so we'll create a simple example
softmax_input = np.array([[1, 2, 3], [4, 5, 6]])
softmax_output = softmax(softmax_input)

# Plot the outputs
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_output)
plt.title("Sigmoid")

plt.subplot(2, 2, 2)
plt.plot(x, tanh_output)
plt.title("Tanh")

plt.subplot(2, 2, 3)
plt.plot(x, relu_output)
plt.title("ReLU")

plt.subplot(2, 2, 4)
plt.bar(range(3), softmax_output[0])
plt.title("Softmax")

plt.tight_layout()
plt.show()