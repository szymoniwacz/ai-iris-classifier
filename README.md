# Iris Classifier — Decision Tree Experiments

## Goal

Understand how a Decision Tree works in practice:
- how it splits data
- how `max_depth` affects performance
- how overfitting appears

This project is focused on learning by experiments, not just theory.

---

## Dataset

- Iris dataset (built into scikit-learn)
- 150 samples
- 3 classes:
  - setosa
  - versicolor
  - virginica

Features:
- sepal length
- sepal width
- petal length
- petal width

---

## Tech Stack

- Python
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## How to run

```bash
git clone https://github.com/szymoniwacz/ai-iris-classifier.git
cd ai-iris-classifier

python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## What I did

### 1. Basic model

- trained DecisionTreeClassifier
- default parameters

### 2. Train / test split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Why:
- simulate real-world scenario
- avoid testing on training data

---

## Max Depth Experiment

### How to run

    python -m src.cli experiment-max-depth

### Example output

    depth=1 | train=0.667 | test=0.667
    depth=2 | train=0.967 | test=0.933
    depth=3 | train=0.983 | test=0.967
    depth=4 | train=0.992 | test=0.933
    depth=5 | train=1.000 | test=0.933
    depth=None | train=1.000 | test=0.933

    Best result:
    - max_depth = 3
    - test accuracy = 0.967

    Interpretation:
    - depth 1 → underfitting
    - depth 3 → best generalization
    - deeper → overfitting

### Plot

![max depth plot](images/max_depth_plot.png)

### Interpretation

| Observation | Meaning |
|---|---|
| depth=1 | model is too simple |
| depth=2 | strong improvement |
| depth >= 3 | perfect test accuracy |
| train keeps increasing | model becomes more complex |

### Conclusion

For this dataset, deeper trees still generalize well.

The Iris dataset is simple, so increasing `max_depth` does not reduce test accuracy here.

However, higher depth increases model complexity and may cause overfitting on more complex datasets.

---

## Project structure

```
iris-classifier/
├── notebook.ipynb
├── README.md
├── requirements.txt
└── images/
    └── tree.png
```

---

## requirements.txt

```
scikit-learn
matplotlib
jupyter
```

---

## What I learned

- how Decision Trees split data
- how to detect overfitting
- why validation is critical
- how model complexity impacts performance

---

## Next steps

- compare with k-NN
- try Logistic Regression
- add confusion matrix
- test on different dataset (Wine / Titanic)

---

## Notes

This is part of a bigger learning path:
- building real ML intuition
- creating GitHub portfolio
- combining backend + AI skills

---

## Author

Szymon Iwacz
