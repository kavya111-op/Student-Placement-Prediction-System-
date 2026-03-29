import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load dataset
    data = pd.read_csv("dataset.csv")

    X = data[['cgpa','skills','internships']]
    y = data['placed']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model