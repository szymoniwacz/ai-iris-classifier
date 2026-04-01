from sklearn.tree import DecisionTreeClassifier


def create_model(max_depth=3):
    return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
