from src.formatters.confusion_matrix_formatter import format_confusion_matrix_result


def test_format_confusion_matrix_result_includes_summary_and_report():
    result = {
        "matrix": [
            [10, 0, 0],
            [0, 9, 1],
            [0, 0, 10],
        ],
        "labels": ["setosa", "versicolor", "virginica"],
        "plot_path": "artifacts/confusion_matrix_plot.png",
        "classification_report": "precision recall f1-score support",
    }

    formatted = format_confusion_matrix_result(result)

    assert "=== Confusion Matrix Experiment ===" in formatted
    assert "Correct predictions: 29/30" in formatted
    assert "1 sample(s) of 'versicolor' were predicted as 'virginica'" in formatted
    assert "Classification report:" in formatted
    assert "Plot saved to: artifacts/confusion_matrix_plot.png" in formatted


def test_format_confusion_matrix_result_reports_no_confusions_when_diagonal_only():
    result = {
        "matrix": [
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
        ],
        "labels": ["setosa", "versicolor", "virginica"],
        "plot_path": "artifacts/confusion_matrix_plot.png",
        "classification_report": "precision recall f1-score support",
    }

    formatted = format_confusion_matrix_result(result)

    assert "- No class confusions detected" in formatted
