from pathlib import Path

import joblib


MODEL_PATH = Path("artifacts/model.joblib")


def load_payload():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run training first: python -m src.cli train"
        )

    return joblib.load(MODEL_PATH)


def predict(sample):
    payload = load_payload()

    model = payload["model"]
    class_names = payload["class_names"]

    prediction = model.predict([sample])
    predicted_class_index = int(prediction[0])

    return class_names[predicted_class_index]