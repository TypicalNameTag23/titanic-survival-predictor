import pandas
import sqlite3
import os
import sys

import joblib
import argparse
from utils import clean_data
from utils import train_model

def load_data_error(message):
    print(message)
    sys.exit(1)

print()

#
# Load training data into data frame
#

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "Path to CSV or SQLite .db file")
parser.add_argument("--split", action = "store_true", help = "Split training data into train / test sets to evaluate accuracy during execution")
parser.add_argument("--estimators", type = int, default = 100, help = "Number of estimator in the random forest classifier")
parser.add_argument("--output-file", default = "models/titanic_survivor_model.joblib", help = "Path to save the trained model")
args = parser.parse_args()
filename = args.filename

# Load data into data frame
if not os.path.exists(filename):
    load_data_error(f"File not found: {filename}")
elif filename.endswith(".csv"):
    try:
        data_frame = pandas.read_csv(filename)
    except Exception as e:
        load_data_error(f"Error while reading data from {filename}: {e}")
elif filename.endswith(".db"):
    try:
        connection = sqlite3.connect(filename)
        data_frame = pandas.read_sql_query("SELECT * FROM passangers", connection)
        connection.close()
    except Exception as e:
        load_data_error(f"Error while reading data from {filename}: {e}")
else:
    load_data_error(f"Unsupported file type: {filename}. Training data must be CSV of Sqlite .db file")


print(f"Succesfully loaded {len(data_frame)} rows from {filename}:")
print(data_frame.head())
print()

data_frame = clean_data(data_frame)

model = train_model(data_frame, args.split)

joblib.dump(model, args.output_file)
print(f"Model saved to {args.output_file}")
