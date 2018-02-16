import csv
import sys

def log_titles(filename, titles):
    try:
        f = open(filename, 'wt')
        writer = csv.writer(f)
        writer.writerow(titles)
    except Exception as e:
        print("Cannot log information ", e)


def log_append(filename, row):
    try:
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("Cannot log information ", e)
