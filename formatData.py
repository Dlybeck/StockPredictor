import json
import time

start_time = time.time()

file = open('sp500.json')
data = json.load(file)

practice_inputs = []
practice_labels = []

#Go through every company
for company in data:
    all_days = company['data']
    
    #excluding the last 504 days, create an input for each range of days (sliding window method)
    for i in range(0, len(all_days)-(504+5)):
        #add to this input
        input = []
        for j in range(i, i+504):
            input.append(all_days[j])
        
        label = []
        for j in range(i+504, i+504+5):
            label.append(all_days[j])
            
        practice_inputs.append(input)
        practice_labels.append(label)
    
    
with open("inputs.json", "w") as outfile:
    json.dump(practice_inputs, outfile, indent=4)
    
with open("labels.json", "w") as outfile:
    json.dump(practice_labels, outfile, indent=4)

end_time = time.time()
print(f"Finished in {round(end_time-start_time, 1)}")

print(len(practice_inputs))
print(len(practice_labels))