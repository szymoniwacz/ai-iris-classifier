from pathlib import Path

import joblib


MODEL_PATH = Path("artifacts/model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run training first: python -m src.cli train"
        )

    return joblib.load(MODEL_PATH)


def predict(sample):
    model = load_model()
    prediction = model.predict([sample])
    return int(prediction[0])