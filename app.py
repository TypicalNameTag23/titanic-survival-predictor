from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas
import io
from utils import clean_data
from utils import train_model
from flask_cors import CORS

 
# Create app instance
app = Flask(__name__)
CORS(app)

model_path = os.environ.get("MODEL_FILE", "models/titanic_survivor_model.joblib")
data_path = os.environ.get("DATA_FILE", "data-sets/titanic.csv")

model = joblib.load(model_path)

@app.route("/predict", methods = ["POST"])
def predict():
    
    #
    # Acceptable JSON format:
    # 
    # {
    #   "Pclass": 3
    #   "Sex": "male"
    #   "Age": 22
    #   "SibSp": 1
    #   "Parch": 0
    #   "Fare": 7.25
    #   "Embarked": "S"
    # }
    #

    data = request.get_json()

    # Return if request is malformed
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    
    # Map male -> 0, female -> 1
    if data["Sex"] == "male":
        sex = 0
    else:
        sex = 1

    # Map S -> 0, C -> 1, Q -> 2
    if data["Embarked"] == "S":
        embarked = 0
    elif data["Embarked"] == "C":
        embarked = 1
    else:
        embarked = 2

    # Build features array
    features = np.array([[
        data["Pclass"],
        sex,
        data["Age"],
        data["SibSp"],
        data["Parch"],
        data["Fare"],
        embarked
    ]])

    probability = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    return jsonify({"predicted_survival_chance": round(float(probability), 2),
                    "prediction": int(prediction)})

@app.route("/retrain", methods=["POST"])
def retrain():
    global model

    if not request.data:
        return jsonify({"error": "No file provided"}), 400
    
    try:
        raw = request.data.decode("utf-8")
        new_data_frame = pandas.read_csv(io.StringIO(raw))
    except Exception as e:
        return jsonify({"error": f"Could not parse request body as CSV: {e}"}), 400

    existing_data_frame = pandas.read_csv(data_path)
    data_frame = pandas.concat([existing_data_frame, new_data_frame], ignore_index = True)

    data_frame = clean_data(data_frame)

    model = train_model(data_frame)

    joblib.dump(model, model_path)
    data_frame.to_csv(data_path, index = False)

    return jsonify({"message": f"Model retrained on {len(data_frame)} rows"})



if __name__ == "__main__":
    app.run()
