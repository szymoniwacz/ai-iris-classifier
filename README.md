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
git clone https://github.com/YOUR_USERNAME/iris-classifier.git
cd iris-classifier

python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

jupyter notebook
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
