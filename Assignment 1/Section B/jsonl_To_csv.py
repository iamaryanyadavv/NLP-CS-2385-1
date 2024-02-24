import json
import csv

# Open the JSONL file and the output CSV file
with open('OurDataset.jsonl', 'r') as jsonl_file, open('OurDataset.csv', 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)
    
    # Initialize a variable to track whether headers have been written
    headers_written = False
    
    for line in jsonl_file:
        # Load the JSON object from the current line
        data = json.loads(line)
        
        # If headers haven't been written yet, write them based on the keys of the first JSON object
        if not headers_written:
            headers = data.keys()
            csv_writer.writerow(headers)
            headers_written = True
        
        # Write the data values to the CSV file
        csv_writer.writerow(data.values())
