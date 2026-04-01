from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.model_factory import create_model


def run():
    X_train, X_test, y_train, y_test = load_data()

    depths = [1, 2, 3, 4, 5, None]
    results = []

    for depth in depths:
        model = create_model(max_depth=depth)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        results.append(
            {
                "depth": depth,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

    return results