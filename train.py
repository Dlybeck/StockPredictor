import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, BatchNormalization, Flatten, Dense, Input
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
with open('training_inputs.json', 'r') as f:
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
with open('training_labels.json', 'r') as f:
    labels = np.array(json.load(f))

print(f"{current_time()}: Data loaded. Normalizing data...")

# Normalize the data
scaler = MinMaxScaler()
inputs_reshaped = inputs.reshape(-1, inputs.shape[-1])
inputs_scaled = scaler.fit_transform(inputs_reshaped)
inputs = inputs_scaled.reshape(-1, 104, 6)  # 104 time steps, 6 features

# Normalize the labels
scaler_labels = MinMaxScaler()
labels_reshaped = labels.reshape(-1, labels.shape[-1])
labels_scaled = scaler_labels.fit_transform(labels_reshaped)
labels = labels_scaled.reshape(-1, 4)  # 4 target values


# Print shapes for debugging
print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

print(f"{current_time()}: Splitting data into training and testing sets...")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print(f"{current_time()}: Defining the Conv1D + LSTM model...")

# Define the custom loss function
def scaled_mae(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return mae * 100

# Define the model
model = Sequential([
    Input(shape=(104, 6)),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(4, activation='linear')  # Output layer with 4 units for 4 target values
])

print(f"{current_time()}: Compiling the model...")

# Summary of the model
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00000075), loss=scaled_mae, metrics=['mae'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print(f"{current_time()}: Starting training...")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

print(f"{current_time()}: Training completed. Saving the model...")

# Save the model
model.save('stock_conv1d_lstm_model3.h5')

print(f"{current_time()}: Model saved. Plotting training history...")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"{current_time()}: Training history plotted.")
