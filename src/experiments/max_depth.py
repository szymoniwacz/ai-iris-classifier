from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.model_factory import create_model


def run():
    X_train, X_test, y_train, y_test = load_data()

    depths = [1, 2, 3, 4, 5, None]

    results = []

    for depth in depths:
        model = create_model(max_depth=depth)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results.append(
            {
                "depth": depth,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

    # --- chart ---
    depths_plot = [r["depth"] if r["depth"] is not None else 6 for r in results]
    train_scores = [r["train_accuracy"] for r in results]
    test_scores = [r["test_accuracy"] for r in results]

    plt.figure()
    plt.plot(depths_plot, train_scores, marker="o", label="train")
    plt.plot(depths_plot, test_scores, marker="o", label="test")

    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Decision Tree: max_depth vs accuracy")
    plt.legend()

    plt.savefig("artifacts/max_depth_plot.png")
    plt.close()

    return results
