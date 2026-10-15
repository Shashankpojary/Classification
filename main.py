"""
Project 2: Data Classification Using AI
A beginner-friendly supervised-learning classification example using scikit-learn's Iris dataset.
Run: python main.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1. Load and understand the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("DATA CLASSIFICATION USING AI")
    print("-" * 40)
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {', '.join(iris.target_names)}")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Train a simple supervised classification algorithm
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 4. Test and evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # 5. Predict a new sample (sepal length, sepal width, petal length, petal width)
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)[0]
    print("\nExample new flower:")
    print(f"Features: {sample[0]}")
    print(f"Predicted class: {iris.target_names[prediction]}")

if __name__ == "__main__":
    main()
