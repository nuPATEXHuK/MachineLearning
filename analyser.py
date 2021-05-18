import os
import csv

current_path = os.getcwd()
filename = os.path.join(current_path, "result.csv")

errors = 0
i = 0

with open(filename) as File:
    reader = csv.reader(File, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        if row[0] != row[1]:
            errors += 1
        i += 1
print("Errors: {} of {}".format(errors, i))
print("Percent of errors: {}%".format(round(errors/i * 100, 2)))
