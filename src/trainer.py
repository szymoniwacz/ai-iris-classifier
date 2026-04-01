from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.model_factory import create_model

MODEL_PATH = Path("artifacts/model.joblib")


def train():
    X_train, X_test, y_train, y_test = load_data()

    model = create_model(max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    iris = load_iris()
    payload = {"model": model, "class_names": iris.target_names.tolist()}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    return {"accuracy": accuracy, "model_path": str(MODEL_PATH)}
