import sys

from src.predictor import predict
from src.trainer import train


def validate_predict_args(args):
    if len(args) != 4:
        raise ValueError(
            "Predict requires exactly 4 numeric values: "
            "sepal_length sepal_width petal_length petal_width"
        )

    try:
        return [float(value) for value in args]
    except ValueError as error:
        raise ValueError("All prediction inputs must be numeric.") from error


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.cli train")
        print("  python -m src.cli predict 5.1 3.5 1.4 0.2")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        result = train()
        print(f"Training completed. Accuracy: {result['accuracy']:.3f}")
        print(f"Model saved to: {result['model_path']}")

    elif command == "predict":
        try:
            sample = validate_predict_args(sys.argv[2:6])
            result = predict(sample)
            print(f"Predicted class: {result}")
        except ValueError as error:
            print(f"Input error: {error}")
            sys.exit(1)
        except FileNotFoundError as error:
            print(error)
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict")
        sys.exit(1)


if __name__ == "__main__":
    main()