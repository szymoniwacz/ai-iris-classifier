import pytest

from src.cli import validate_predict_args


def test_validate_predict_args_converts_values_to_float():
    result = validate_predict_args(["5.1", "3.5", "1.4", "0.2"])

    assert result == [5.1, 3.5, 1.4, 0.2]


def test_validate_predict_args_requires_exactly_four_values():
    with pytest.raises(ValueError, match="exactly 4 numeric values"):
        validate_predict_args(["5.1", "3.5", "1.4"])


def test_validate_predict_args_rejects_non_numeric_values():
    with pytest.raises(ValueError, match="must be numeric"):
        validate_predict_args(["5.1", "oops", "1.4", "0.2"])
