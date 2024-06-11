import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np

model = tf.keras.models.load_model('stock_lstm_model.h5')

with open('100weeks.json', 'r') as f:
    data = json.load(f)


# Extract the data for a specific stock, e.g., "MMM"
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
    mmm_array_scaled = mmm_array_scaled.reshape(1, 100, 6)  # Reshape to (1, 100, 6)

    # Check the shape of the scaled data
    #print("Scaled shape:", mmm_array_scaled.shape)  # Should be (1, 100, 6)

    # Make predictions
    return model.predict(mmm_array_scaled).tolist()
    #print("Predictions:", predictions)



#create json file of predictions for every company
with open('sp500.json', 'r') as f:
    sp500 = json.load(f)
    
    
all_predicts = {}    
for comp in sp500:
    print(comp['symbol'])
    #Only predict if there's enough data
    if(len(data[comp['symbol']]['data']) == 100):
        all_predicts[comp['symbol']] = predict(comp['symbol'])
    
with open("predictions.json", "w") as outfile:
    json.dump(all_predicts, outfile, indent=4)
