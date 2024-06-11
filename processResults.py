import json

# Load the data
with open('100weeks.json', 'r') as f:
    data = json.load(f)
    
print(len(data[0]['data']))