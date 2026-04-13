from pathlib import Path

from src import predictor, trainer


def test_train_creates_model_artifact_and_returns_metadata(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"

    monkeypatch.setattr(trainer, "MODEL_PATH", Path(model_path))

    result = trainer.train()

    assert model_path.exists()
    assert result["model_path"] == str(model_path)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_predict_returns_known_class_after_training(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"

    monkeypatch.setattr(trainer, "MODEL_PATH", Path(model_path))
    monkeypatch.setattr(predictor, "MODEL_PATH", Path(model_path))

    trainer.train()

    predicted_class = predictor.predict([5.1, 3.5, 1.4, 0.2])

    assert predicted_class in {"setosa", "versicolor", "virginica"}
