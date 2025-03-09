from sklearn.svm import SVC, SVR, LinearSVC
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np



def find_best_params_grid_search(x_train, y_train):
    param_grid = {"C": range(1, 10, 1), "tol": [0.0001, 0.0002, 0, 0.02, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "penalty": ["l2", "l1"]}
    grid_search = RandomizedSearchCV(LinearSVC(), param_grid, cv=5, verbose=1)
    grid_search.fit(x_train, y_train)
    best_C = grid_search.best_params_["C"]
    gamma = grid_search.best_params_["tol"]
    penalty = grid_search.best_params_["penalty"]
    #kernel = grid_search.best_params_["kernel"]

    return best_C, gamma, penalty #, kernel


def main():
    # Read the training data
    df = pd.read_csv(r"titanic/train.csv")
    df = df.dropna(subset=["Age"])
    print(len(df))

    # Set 20% for testing and 80% for training/validation
    x_test = df.sample(frac=0.15, random_state=4)
    x_train = df.drop(x_test.index)  # just drop the indexes from x_test

    # Get the labels for the train and test portions
    y_test = x_test["Survived"]
    y_train = x_train["Survived"]

    features = ["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Parch", 'Embarked', "SibSp", "Fare"]

    # set the x_train and x_test features properly, drop some if needed, and leave out the y_train column (Survived)
    x_train = x_train.drop(features, axis=1)
    sex = {"male": 0, "female": 1}
    embark = {"S":0, "Q":1, "C":2}
    x_train["Sex"] = x_train["Sex"].map(sex)
    #x_train["Embarked"] = x_train["Embarked"].map(embark)
    
    # get the mean of the age column
    x_train["Age"] = pd.to_numeric(x_train["Age"])
    x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())
    #x_train["Embarked"] = x_train["Embarked"].fillna(4)
    #print(x_train.columns)
    #print(x_train["Embarked"].isnull().values.any())


    x_test = x_test.drop(features, axis=1)
    sex = {"male": 0, "female": 1}
    embark = {"S":0, "Q":1, "C":2}
    x_test["Sex"] = x_test["Sex"].map(sex)
    #x_test["Embarked"] = x_test["Embarked"].map(embark)

    x_test["Age"] = pd.to_numeric(x_test["Age"])
    x_test["Age"] = x_test["Age"].fillna(x_test["Age"].mean())
    #x_test["Embarked"] = x_test["Embarked"].fillna(4)

    # Keep the most common name convention
    X = x_train
    y = y_train

    best_C, gamma, penalty = find_best_params_grid_search(X, y)
    print(f"Best C = {best_C}, and best gamma = {gamma}, and penalty = {penalty}")

    # Use the best hyperparameters, found by using RandomizedSearchCV, with CV=5
    model = LinearSVC(C=best_C, tol=gamma, penalty=penalty)
    model = model.fit(X, y)

    # Test on the test subset we set apart at the beginning
    y_pred = model.predict(x_test)
    # Print some stats for both classes
    print(classification_report(y_test, y_pred))

    # Now using all the data
    X = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])
    model = LinearSVC(C=best_C, tol=gamma, penalty=penalty)
    model= model.fit(X, y)

    # Loading kaggle test set
    kaggle_test = pd.read_csv(r"titanic/test.csv")
    kaggle_test = kaggle_test.drop(["PassengerId", "Name", "Ticket", "Cabin", 'Embarked', "SibSp", "Fare"], axis=1)
    sex = {"male": 0, "female": 1}
    #embark = {"S":0, "Q":1, "C":2}
    kaggle_test["Sex"] = kaggle_test["Sex"].map(sex)
    #kaggle_test["Embarked"] = kaggle_test["Embarked"].map(embark)

    kaggle_test["Age"] = pd.to_numeric(kaggle_test["Age"])
    kaggle_test["Age"] = kaggle_test["Age"].fillna(kaggle_test["Age"].mean())

    # Predict with the random forest classifier
    y_pred = model.predict(kaggle_test)

    # Save the predictions to a csv file
    prediction = pd.DataFrame(y_pred, columns=['Survived']).to_csv('titanic_predictions.csv')

main()

