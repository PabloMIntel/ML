import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np

def plot_acc(range, scores):
    plt.plot(range, scores, marker='o')
    plt.xlabel("Tree Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation accuracy vs Tree Depth")
    plt.show()

def find_best_depth_Tree_grid_search(x_train, y_train):
    param_grid = {"max_depth": range(1,10)}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5, verbose=2)
    grid_search.fit(x_train, y_train)
    best_depth = grid_search.best_params_["max_depth"]
    return best_depth

def find_best_depth_Tree_CV(x_train, y_train):
    scores = []
    for depth in range(1, 10):
        d_tree = DecisionTreeClassifier(max_depth=depth)
        scores.append(np.mean(cross_val_score(d_tree, x_train, y_train, cv=5)))

    best_depth = scores.index(max(scores))
    plot_acc(range(1, 10), scores)

    return (best_depth+1)

def find_best_params_Forest_grid_search(x_train, y_train):
    param_grid = {"n_estimators":range(70, 130), "max_depth": range(1,20), "min_samples_split": range(2,15), 
                  "max_features": ["sqrt", "log2", None]}
    grid_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1)
    grid_search.fit(x_train, y_train)
    best_depth = grid_search.best_params_["max_depth"]
    n_estimators = grid_search.best_params_["n_estimators"]
    max_features = grid_search.best_params_["max_features"]
    min_split = grid_search.best_params_["min_samples_split"]

    return n_estimators, best_depth, min_split, max_features
    

def main():
    # Read the training data
    df = pd.read_csv(r"titanic/train.csv")
    print(len(df))

    # Set 20% for testing and 80% for training/validation
    x_test = df.sample(frac=0.20, random_state=42)
    x_train = df.drop(x_test.index)  # just drop the indexes from x_test

    # Get the labels for the train and test portions
    y_test = x_test["Survived"]
    y_train = x_train["Survived"]

    # set the x_train and x_test features properly, drop some if needed, and leave out the y_train column (Survived)
    x_train = x_train.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    sex = {"male": 0, "female": 1}
    embark = {"S":0, "Q":1, "C":2}
    x_train["Sex"] = x_train["Sex"].map(sex)
    x_train["Embarked"] = x_train["Embarked"].map(embark)

    x_test = x_test.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    sex = {"male": 0, "female": 1}
    embark = {"S":0, "Q":1, "C":2}
    x_test["Sex"] = x_test["Sex"].map(sex)
    x_test["Embarked"] = x_test["Embarked"].map(embark)

    # Keep the most common name convention
    X = x_train
    y = y_train

    # Find the best depth, you can use grid search or a more simple CV approach
    n_estimators, best_depth, min_samples_split, max_features = find_best_params_Forest_grid_search(X, y)
    print("Best fit found at depth:", best_depth)
    print("Best fit found at n_estimators:", n_estimators)
    print("Best fit found at min_samples_split:", min_samples_split)
    print("Best fit found at max_features:", max_features)

    # Use the best hyperparameters, found by using RandomizedSearchCV, with CV=5
    d_tree = RandomForestClassifier(n_estimators=n_estimators, max_depth=best_depth,
                                    min_samples_split=min_samples_split, max_features=max_features)
    d_tree = d_tree.fit(X, y)

    # Test on the test subset we set apart at the beginning
    y_pred = d_tree.predict(x_test)
    # Print some stats for both classes
    print(classification_report(y_test, y_pred))

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy for the test set:", metrics.accuracy_score(y_test, y_pred))

    # Now using all the data
    X = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])
    d_tree = RandomForestClassifier(n_estimators=n_estimators, max_depth=best_depth,
                                    min_samples_split=min_samples_split, max_features=max_features)
    d_tree= d_tree.fit(X, y)

    # Read the kaggle test dataset and drop the same columns as training and test dataset
    kaggle_test = pd.read_csv(r"titanic/test.csv")
    kaggle_test = kaggle_test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    sex = {"male": 0, "female": 1}
    embark = {"S":0, "Q":1, "C":2}
    kaggle_test["Sex"] = kaggle_test["Sex"].map(sex)
    kaggle_test["Embarked"] = kaggle_test["Embarked"].map(embark)

    # Predict with the random forest classifier
    y_pred = d_tree.predict(kaggle_test)

    # Save the predictions to a csv file
    prediction = pd.DataFrame(y_pred, columns=['Survived']).to_csv('titanic_predictions.csv')

main()