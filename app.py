from flask import Flask, request, jsonify
import joblib
import numpy as np
import argparse
 
# Create app instance
app = Flask(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-file", default = "models/titanic_survivor_model.joblib", help = "Path to joblib model file")
args = parser.parse_args()

model = joblib.load(args.model_file)

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



if __name__ == "__main__":
    app.run()