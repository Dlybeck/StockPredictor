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
    data = OrderedDict(list(data.items()))
    
    for k, v in data.items():
        v["Change"] = round((v["Close"] - v["Open"]) / v["Close"], 3)
    
    return data

def get_company_basics(symbol):
    # Get company info
    info = yf.Ticker(symbol).info

    all_data = {}

    # Remove unwanted info
    to_keep = {'symbol', 'longName'}
    for key in info:
        if key in to_keep:
            all_data[key] = info[key]
    
    data = OrderedDict(reversed(list(all_data.items())))
    
    return data

def get_sp500_companies():
    # Get all S&P 500 company symbols from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)

    # The first table on the page contains the list of S&P 500 companies
    sp500_table = table[0]

    # So it works with yfinance
    sp500_symbols = [symbol.replace('.', '-') for symbol in sp500_table['Symbol'].tolist()]

    return sp500_symbols

sp500 = get_sp500_companies()

all_data = []
processed_companies = []

for comp in sp500:
    print(f"Processing {comp}...")
    data = {}
    basics = get_company_basics(comp)
    for key in basics:
        data[key] = basics[key]

    if 'longName' in basics:
        processed_companies.append(f"{comp} - {basics['longName']}")
    else:
        processed_companies.append(comp)
    
    history = get_company_history(comp)

    #append the stock info for every day of the current company
    day_Data = []
    for date in history:
        day_Data.append(history[date])
        #add the date in the day's info for reference
        history[date]['Date'] = date
    
    data['data'] = day_Data
    
    all_data.append(data)

with open("sp500.json", "w") as outfile:
    json.dump(all_data, outfile, indent=4)

end_time = time.time()
print(f"Finished in {round(end_time-start_time, 1)} seconds")

# Print the list of processed companies
print(len(processed_companies), "    companies Processed")