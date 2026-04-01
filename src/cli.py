import sys

from src.experiments.max_depth import run as run_max_depth_experiment
from src.predictor import predict
from src.trainer import train


AVAILABLE_COMMANDS = (
    "train",
    "experiment-max-depth",
    "predict",
)


def print_usage():
    print("Usage:")
    print("  python -m src.cli train")
    print("  python -m src.cli experiment-max-depth")
    print("  python -m src.cli predict 5.1 3.5 1.4 0.2")


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


def handle_train(_args):
    result = train()
    print(f"Training completed. Accuracy: {result['accuracy']:.3f}")
    print(f"Model saved to: {result['model_path']}")


def handle_experiment_max_depth(_args):
    results = run_max_depth_experiment()
    print("Experiment completed.")
    print("Plot saved to: artifacts/max_depth_plot.png")

    for row in results:
        print(
            f"depth={row['depth']}, "
            f"train={row['train_accuracy']:.3f}, "
            f"test={row['test_accuracy']:.3f}"
        )


def handle_predict(args):
    try:
        sample = validate_predict_args(args)
        result = predict(sample)
        print(f"Predicted class: {result}")
    except ValueError as error:
        print(f"Input error: {error}")
        sys.exit(1)
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)


COMMAND_HANDLERS = {
    "train": handle_train,
    "experiment-max-depth": handle_experiment_max_depth,
    "predict": handle_predict,
}


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    handler = COMMAND_HANDLERS.get(command)

    if handler is None:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(AVAILABLE_COMMANDS)}")
        print_usage()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()