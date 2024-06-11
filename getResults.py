import yfinance as yf
import json
from collections import OrderedDict
import pandas as pd

def get_company_history(symbol):
    data = yf.download(symbol, start="2000-01-01", progress=False, interval='1wk')

    # Convert DataFrame to dict and then to JSON
    data.index = data.index.astype(str)  # Convert index to string
    data = data.to_dict(orient='index')
    
    # Put recent ones on top of json
    data = OrderedDict(list(data.items()))
    
    return data

#Get the most recent 100 weeks worth of data for every company
def get_sp500_companies():
    # Get all S&P 500 company symbols from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)

    # The first table on the page contains the list of S&P 500 companies
    sp500_table = table[0]

    # So it works with yfinance
    sp500_symbols = [symbol.replace('.', '-') for symbol in sp500_table['Symbol'].tolist()]

    return sp500_symbols

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

sp500 = get_sp500_companies()

all_data = {}
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

    #append the stock info for every week of the current company
    counter = 0
    day_Data = []
    
    #Add date to the data for the most recent 100 weeks
    for date in history: 
        if(counter < 100):
            day_Data.append(history[date])
            #add the date in the day's info for reference
            #history[date]['Date'] = date
            counter += 1
    
    data['data'] = day_Data
    
    all_data[basics['symbol']] = data

with open("100weeks.json", "w") as outfile:
    json.dump(all_data, outfile, indent=4)