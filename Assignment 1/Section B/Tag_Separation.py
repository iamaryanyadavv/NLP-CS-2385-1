# import csv
# import ast

# input_file_path = 'Tags.csv'
# output_file_path = 'Tags_Modified.csv'

# def process_row(row):
#     try:
#         # Debugging: Print the row to see what it looks like
#         print("Row before processing:", row[0])
#         array = ast.literal_eval(row[0])  # Attempt to evaluate the string as a Python literal
#         modified_array = [sub_array[2] for sub_array in array]
#         return modified_array
#     except ValueError as e:
#         print(f"Error processing row: {e}")
#         return []  # Return an empty list or handle the error as appropriate

# with open(input_file_path, mode='r', encoding='utf-8') as infile, \
#      open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
    
#     for row in reader:
#         modified_array = process_row(row)
#         writer.writerow([modified_array])

# print("Process completed. The modified arrays have been written to", output_file_path)

import csv
import ast

input_file_path = 'Tags.csv'
output_file_path = 'Tags_Modified2.csv'

def process_row(row):
    try:
        print("Row before processing:", row[0])
        array = ast.literal_eval(row[0])  # Attempt to evaluate the string as a Python literal
        category_1_tags = {'NN', 'NNP', 'RB', 'DT', 'IN', 'CD', 'CC', 'JJ', 'PRP', 'CD', 'NNP', 'VB', 'MD', 'WDT'}
        category_2_tags = {'B-PER', 'I-PER', 'B-GEO', 'I-GEO', 'O', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-NORP', 'I-NORP', 'B-DATE', 'I-DATE', 'B-PRODUCT', 'I-PRODUCT'}

        category_1_array = []
        category_2_array = []

        for sub_array in array:
            if sub_array[2] in category_1_tags:
                category_1_array.append(sub_array[2])
            elif sub_array[2] in category_2_tags:
                category_2_array.append(sub_array[2])
            else:
                print(f"Tag '{sub_array[2]}' not found in any category")

        return category_1_array, category_2_array
    except ValueError as e:
        print(f"Error processing row: {e}")
        return [], []  # Return two empty lists or handle the error as appropriate

with open(input_file_path, mode='r', encoding='utf-8') as infile, \
     open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        category_1_array, category_2_array = process_row(row)
        writer.writerow([category_1_array, category_2_array])

print("Process completed. The modified arrays have been written to", output_file_path)
