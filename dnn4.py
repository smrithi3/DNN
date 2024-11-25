# 4. Design and implement a CNN model (with 4+ layers of convolutions) to classify multi category image datasets. Use the concept of regularization and dropout while designing the CNN model. Use the Fashion MNIST datasets. Record the Training accuracy and Test accuracy corresponding to the following architectures:
    
    #a. Base Model
    
    #b. Model with L1 Regularization
    
    #c. Model with L2 Regularization
    
    #d. Model with Dropout

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data to include the channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


def create_base_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Use Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_l1_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Use Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_l2_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Use Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_dropout_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Use Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout layer added
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return train_acc, test_acc

# Create and evaluate base model
base_model = create_base_model()
base_train_acc, base_test_acc = train_and_evaluate_model(base_model, x_train, y_train, x_test, y_test)

# Create and evaluate L1 model
l1_model = create_l1_model()
l1_train_acc, l1_test_acc = train_and_evaluate_model(l1_model, x_train, y_train, x_test, y_test)

# Create and evaluate L2 model
l2_model = create_l2_model()
l2_train_acc, l2_test_acc = train_and_evaluate_model(l2_model, x_train, y_train, x_test, y_test)

# Create and evaluate Dropout model
dropout_model = create_dropout_model()
dropout_train_acc, dropout_test_acc = train_and_evaluate_model(dropout_model, x_train, y_train, x_test, y_test)

results = {
    "Base Model": {"Train Accuracy": base_train_acc, "Test Accuracy": base_test_acc},
    "L1 Regularization": {"Train Accuracy": l1_train_acc, "Test Accuracy": l1_test_acc},
    "L2 Regularization": {"Train Accuracy": l2_train_acc, "Test Accuracy": l2_test_acc},
    "Dropout": {"Train Accuracy": dropout_train_acc, "Test Accuracy": dropout_test_acc}
}

for model_name, accuracies in results.items():
    print(f"{model_name} - Train Accuracy: {accuracies['Train Accuracy']:.4f}, Test Accuracy: {accuracies['Test Accuracy']:.4f}")