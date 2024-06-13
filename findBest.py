import json
import yfinance as yf

with open('predictions2.json', 'r') as f:
    data = json.load(f)
    
increases = []
#for every company
for comp, prediction in data.items():
    total_change = 1
    #find the 4 week percent change
    for change in prediction[0]:
        total_change *= (change/100)+1

    #add company to list
    increases.append((total_change, comp, prediction))

increases.sort()
increases.reverse()

i = 0
best = []
while (i < 5):
    best.append(increases[i])
    i+=1

print(f"I predict the best 5 stocks to invest in over the next 4 weeks are:")
for (change, comp, prediction) in best:
    full_name = yf.Ticker(comp).info['longName']
    print(f"{full_name} ({round(change, 4)}% change):\n   Week 1: {round(prediction[0][0], 4)}% change\n   Week 2: {round(prediction[0][1], 4)}% change\n   Week 3: {round(prediction[0][2], 4)}% change\n   Week 4: {round(prediction[0][3], 4)}% change")