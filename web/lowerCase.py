import csv

input_file = 'static/animals.csv'
output_file = 'lowercase_animals.csv'

with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = [row for row in reader]

lowercase_data = [[cell.lower() for cell in row] for row in data]

with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(lowercase_data)

print(f"Converted data has been saved to {output_file}.")