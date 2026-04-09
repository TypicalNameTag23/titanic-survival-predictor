from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def clean_data(data_frame):
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
    
    return data_frame

def train_model(data_frame, estimators=100, split=None):
    # Initialize the model
    model = RandomForestClassifier(n_estimators = estimators, random_state = 617)

    # Seperate into features data frame and target data frame
    features = data_frame[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    target = data_frame["Survived"]

    # Train the model
    if split:
        # If split evaluate the model after training
        split = train_test_split(features, target, test_size=0.2, random_state = 617)
        features_train, features_test, target_train, target_test = split
        model.fit(features_train, target_train)
        predictions = model.predict(features_test)
        accuracy = accuracy_score(target_test, predictions)
    else:
        model.fit(features, target)

    print("Succesfully trained model")
    if (split):
        print(f"Test accuracy: {accuracy:.2%}")
    print()

    return model