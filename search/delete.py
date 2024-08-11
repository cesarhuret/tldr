import json

# Read the JSON file
with open('search/events.json', 'r') as f:
    file = f.read()
    data = json.loads(file)

# Delete the properties
for event in data['allEvents']:
    if 'slug' in event:
        del event['slug']

# Write the modified data back to the file
with open('/home/cesar/tldr/search/events.json', 'w') as file:
    json.dump(data, file)
