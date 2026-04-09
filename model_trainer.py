import pandas
import sqlite3
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import argparse

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
parser.add_argument("--output-file", default = "titanic_survivor_model.joblib", help = "Path to save the trained model")
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


#
# Clean data
#

# Remove unneeded features
data_frame = data_frame[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

# Map male -> 0, female -> 1
data_frame["Sex"] = data_frame["Sex"].map({"male": 0, "female": 1})

# Map S -> 0, C -> 1, Q -> 2
data_frame["Embarked"] = data_frame["Embarked"].map({"S": 0, "C": 1, "Q": 3})

# Fill empty columns with column average
# Not relevant for our training set, but if Sex has missing values
# filling empty columns with the mean may bias the model toward the
# majority sex. A more neutral imputation strategy such as alternating
# between male and female would be preferred in this case.
for column in data_frame.columns:
    if data_frame[column].dtype != "object":
        data_frame[column] = data_frame[column].fillna(data_frame[column].mean())

# Seperate into features data frame and target data frame
features = data_frame[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
target = data_frame["Survived"]


#
# Train model
#

# Initialize the model
model = RandomForestClassifier(n_estimators = args.estimators, random_state = 617)

# Train the model
if args.split:
    # If split evaluate the model after training
    split = train_test_split(features, target, test_size=0.2, random_state = 617)
    features_train, features_test, target_train, target_test = split
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
else:
    model.fit(features, target)

print("Succesfully trained model")
if (args.split):
    print(f"Test accuracy: {accuracy:.2%}")
print()

joblib.dump(model, args.output_file)
print(f"Model saved to {args.output_file}")
