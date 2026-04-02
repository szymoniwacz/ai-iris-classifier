from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data_loader import load_data
from src.model_factory import create_model

ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "confusion_matrix_plot.png"


def run():
    X_train, X_test, y_train, y_test = load_data()

    model = create_model(max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)

    iris = load_iris()
    labels = iris.target_names

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    disp.plot()

    plt.title("Confusion Matrix - Iris Decision Tree")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    return {
        "matrix": matrix,
        "labels": labels,
        "plot_path": str(PLOT_PATH),
    }
