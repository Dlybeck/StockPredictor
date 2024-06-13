import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the model architecture and weights
model = load_model('stock_conv1d_lstm_model3.h5', compile=False)

# Compile the model with the correct loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error', metrics=['mae'])

# Load the data for prediction
with open('100weeks.json', 'r') as f:
    data = json.load(f)

# Define the prediction function
def predict(symbol):
    mmm_data = data[symbol]['data']

    # Convert the JSON data to a NumPy array
    def convert_single_stock_to_array(data):
        numeric_data = []
        for entry in data:
            numeric_data.append([entry['Open'], entry['High'], entry['Low'], entry['Close'], entry['Adj Close'], entry['Volume']])
        return np.array(numeric_data)

    # Convert the data
    mmm_array = convert_single_stock_to_array(mmm_data)

    # Check the shape of the data
    print("Original shape:", mmm_array.shape)  # Should be (100, 6)

    # Normalize the data
    scaler = MinMaxScaler()
    mmm_array_scaled = scaler.fit_transform(mmm_array)

    # Reshape the data for prediction
    mmm_array_scaled = mmm_array_scaled[-52:]  # Use the last 52 weeks for prediction
    mmm_array_scaled = mmm_array_scaled.reshape(1, 52, 6)  # Reshape to (1, 52, 6)

    # Make predictions
    predictions = model.predict(mmm_array_scaled)

    # If you had used a separate scaler for the predictions during training, you would inverse_transform here.
    # For demonstration, let's assume the predictions are already in the original scale, or you have another scaler.
    # predictions_denormalized = scaler_predictions.inverse_transform(predictions)  # This is an example if you had a scaler for predictions

    return predictions.tolist()

# Create json file of predictions for every company
with open('sp500.json', 'r') as f:
    sp500 = json.load(f)

all_predicts = {}
for comp in sp500:
    symbol = comp['symbol']
    print(f"Processing {symbol}...")
    # Only predict if there's enough data
    if len(data[symbol]['data']) == 100:
        all_predicts[symbol] = predict(symbol)

with open("predictions2.json", "w") as outfile:
    json.dump(all_predicts, outfile, indent=4)

print("Predictions saved to predictions2.json.")
