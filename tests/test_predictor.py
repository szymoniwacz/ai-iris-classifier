from pathlib import Path

import pytest

from src import predictor


def test_load_payload_raises_when_model_file_is_missing(tmp_path, monkeypatch):
    missing_model_path = tmp_path / "missing-model.joblib"
    monkeypatch.setattr(predictor, "MODEL_PATH", Path(missing_model_path))

    with pytest.raises(FileNotFoundError, match="Run training first"):
        predictor.load_payload()
