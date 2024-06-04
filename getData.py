import yfinance as yf
import json
from collections import OrderedDict
import pandas as pd
import time
from forex_python.converter import CurrencyRates

start_time = time.time()

# Initialize the currency converter
currency_converter = CurrencyRates()

def get_company_history(symbol):
    data = yf.download(symbol, start="1990-01-01", progress=False)

    # Convert DataFrame to dict and then to JSON
    data.index = data.index.astype(str)  # Convert index to string
    data = data.to_dict(orient='index')
    
    # Put recent ones on top of json
    data = OrderedDict(reversed(list(data.items())))
    
    for k, v in data.items():
        v["Change"] = round((v["Close"] - v["Open"]) / v["Close"], 3)
    
    return data

def get_company_basics(symbol):
    # Get company info
    info = yf.Ticker(symbol).info

    all_data = {}

    # Remove unwanted info
    to_keep = {'symbol', 'longName', 'state', 'industry', 'sector', 'currency'}
    for key in info:
        if key in to_keep:
            all_data[key] = info[key]
    
    data = OrderedDict(reversed(list(all_data.items())))
    
    return data

def convert_to_usd(data, currency):
    if currency == 'USD':
        return data
    
    exchange_rates = []
    for date in data.index:
        try:
            rate = currency_converter.get_rate(currency, 'USD', date)
        except Exception as e:
            print(f"Could not fetch exchange rate for {date}: {e}")
            rate = currency_converter.get_rate(currency, 'USD')
        exchange_rates.append(rate)
    
    exchange_rates = pd.Series(exchange_rates, index=data.index)
    data[['Open', 'High', 'Low', 'Close', 'Adj Close']] = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].mul(exchange_rates, axis=0)
    
    return data

# Read all ticker symbols from the text file
with open("tickers.txt", "r") as file:
    all_symbols = file.read().splitlines()

all_data = []
processed_companies = []

for comp in all_symbols:
    print(f"Processing {comp}...")
    data = {} 
    basics = get_company_basics(comp)
    for key in basics:
        data[key] = basics[key]
        processed_companies.append(comp)
    
    history = get_company_history(comp)
    
    if 'currency' in basics:
        currency = basics['currency']
    else:
        currency = 'USD'
    
    history_df = pd.DataFrame.from_dict(history, orient='index')
    history_converted = convert_to_usd(history_df, currency)
    history = history_converted.to_dict(orient='index')

    for key in history:
        data[key] = history[key]
    
    all_data.append(data)

with open("all_companies.json", "w") as outfile:
    json.dump(all_data, outfile, indent=4)

end_time = time.time()
print(f"Finished in {round(end_time-start_time, 1)} seconds")

# Print the list of processed companies
print("\nList of processed companies:")
print(processed_companies)
print(len(processed_companies))
