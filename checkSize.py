import json

# Load the data
with open('inputs.json', 'r') as f:
    inputs = json.load(f)

print("file loaded")

'''print(len(inputs))
setLengths = {}
smallSetLengths = {}

for set in inputs:
    length = len(set)
    
    #if this length already exists, add to count
    if(length in setLengths.keys()):
        setLengths[length] += 1
    else:
        setLengths[length] = 1
        
print(setLengths)
'''

print(f"A {len(inputs)} long array of an array {len(inputs[0])} long of {len(inputs[0][0])} long hashtables")
