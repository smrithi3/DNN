#7. Implement the standard VGG-16 & 19 CNN architecture model to classify multi category image dataset and check the accuracy.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Resize images to a smaller size for faster training (128x128)
x_train = tf.image.resize(x_train, (128, 128))
x_test = tf.image.resize(x_test, (128, 128))

# Convert labels to one-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Create the VGG-16 model with frozen base layers
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze VGG16 base layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout to reduce overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create the VGG-19 model with frozen base layers
def create_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze VGG19 base layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Compile and train the VGG-16 model
vgg16_model = create_vgg16_model()
vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the VGG-16 model with reduced epochs and batch size
vgg16_history = vgg16_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the VGG-16 model
vgg16_loss, vgg16_accuracy = vgg16_model.evaluate(x_test, y_test)
print(f'VGG-16 Test Loss: {vgg16_loss:.4f}')
print(f'VGG-16 Test Accuracy: {vgg16_accuracy:.2f}')

# Compile and train the VGG-19 model
vgg19_model = create_vgg19_model()
vgg19_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the VGG-19 model with reduced epochs and batch size
vgg19_history = vgg19_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the VGG-19 model
vgg19_loss, vgg19_accuracy = vgg19_model.evaluate(x_test, y_test)
print(f'VGG-19 Test Loss: {vgg19_loss:.4f}')
print(f'VGG-19 Test Accuracy: {vgg19_accuracy:.2f}')