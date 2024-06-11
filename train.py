import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# Function to print current time for better tracking
def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

print(f"{current_time()}: Loading data...")

# Load the data
with open('inputs.json', 'r') as f:
    inputs = json.load(f)

# Convert the input data to a NumPy array
def convert_to_array(data):
    numeric_data = []
    for sample in data:
        numeric_sample = []
        for entry in sample:
            numeric_sample.append([entry['Open'], entry['High'], entry['Low'], entry['Close'], entry['Adj Close'], entry['Volume']])
        numeric_data.append(numeric_sample)
    return np.array(numeric_data)

inputs = convert_to_array(inputs)

# Load the labels directly from the JSON file
with open('labels.json', 'r') as f:
    labels = np.array(json.load(f))

print(f"{current_time()}: Data loaded. Normalizing data...")

# Normalize the data
scaler = MinMaxScaler()
inputs = inputs.reshape(-1, inputs.shape[-1])
inputs = scaler.fit_transform(inputs)
inputs = inputs.reshape(-1, 100, inputs.shape[-1])  # 100 time steps

print(f"{current_time()}: Splitting data into training and testing sets...")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

print(f"{current_time()}: Defining the LSTM model...")

# Define the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(100, 6), return_sequences=True),  # Input shape: (time_steps, features)
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(4)  # Output shape: (batch_size, 4) assuming 4 target values per time step
])

print(f"{current_time()}: Compiling the model...")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0000005), loss='binary_crossentropy')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

print(f"{current_time()}: Starting training...")

# Set up GPU support if available
if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Not using GPU")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

print(f"{current_time()}: Training completed. Saving the model...")

# Save the model
model.save('stock_lstm_model.h5')

print(f"{current_time()}: Model saved. Plotting training history...")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"{current_time()}: Training history plotted.")
