from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.model_factory import create_model


ARTIFACTS_DIR = Path("artifacts")
PLOT_PATH = ARTIFACTS_DIR / "max_depth_plot.png"


def run():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    depths = [1, 2, 3, 4, 5, None]

    train_scores = []
    test_scores = []
    results = []

    for depth in depths:
        model = create_model(max_depth=depth)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        results.append(
            {
                "depth": depth,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )

        print(f"depth={depth} | train={train_acc:.3f} | test={test_acc:.3f}")

    plt.figure()
    plt.plot([str(d) for d in depths], train_scores, label="train")
    plt.plot([str(d) for d in depths], test_scores, label="test")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Decision Tree: max_depth vs accuracy")
    plt.legend()
    plt.savefig(PLOT_PATH)

    return results