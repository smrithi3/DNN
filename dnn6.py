#6. Implement Bidirectional LSTM for sentiment analysis on movie reviews.

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

# Load and preprocess the IMDB dataset
vocab_size = 10000  # Only consider the top 10,000 words
max_length = 200  # Max length of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),  # Return sequences for additional LSTM layer
    Dropout(0.5),  # Dropout layer to reduce overfitting
    Bidirectional(LSTM(64)),  # Additional LSTM layer
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (positive or negative sentiment)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjusted learning rate
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)  # Increased epochs

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")