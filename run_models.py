"""
This is the main file for running all the models in the project.
Author: Jack St Clair and Zachary Toner
Date: 12/20/2019
"""
import util
import numpy as np
import pickle
#import fc_nn

from copy import deepcopy

# sklearn imports
from sklearn import utils
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def runTuneTest(learner, params, X, y):
    """
    This function takes the model to be created, parameters and data. Then it
    computes the scores and returns, for each fold, the best parameters,
    training score and test score.
    """
    best_params = []
    train_scores = []
    test_scores = []
    test_score = 0.0
    best_model = None

    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    for train_index, test_index in skf.split(X, y): # making each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GridSearchCV(learner, params, cv = 3, verbose = 1)
        clf.fit(X_train, y = y_train)
        best_params.append(clf.best_params_)
        train_scores.append(clf.score(X_train, y = y_train))
        new_score = clf.score(X_test, y = y_test)
        test_scores.append(new_score)
        if new_score >= test_score:
            test_score = new_score
            best_model = deepcopy(clf)

    return best_params, train_scores, test_scores, best_model

def run_KNN(X,y):
    """
    This function will take in data and run it on a knn classifier.
    """
    knn_clf = KNeighborsClassifier()
    parameters = {"weights": ["uniform", "distance"], "n_neighbors": [1, 5, 11]}
    best_params, train_scores, test_scores, best_model = runTuneTest(knn_clf, parameters, X, y)
    print("---------------")
    print("KNN")
    print("---------------" + "\n")
    for i in range(len(best_params)):
        print("Fold " + str(i+1) + ":")
        print(str(best_params[i]))
        print("Training Score: " + str(train_scores[i]) + "\n")
    print("Fold, Test Accuracy")
    for i in range(len(test_scores)):
        print(str(i+1) + ", " + str(test_scores[i]))
    print("\n")

    #print("Testing best model...")
    #modelscore = best_model.score(X, y)
    #print(str(modelscore))
    print("best model params:")
    print(str(best_model.get_params))

    return(best_model)

def run_random_forest(X, y):
    """
    This function will take in data and run it on a random forest classifier.
    """
    clf = RandomForestClassifier(n_estimators = 200)
    parameters = {"max_features": [1/100.0, 1/10.0, "sqrt"]}
    best_params, train_scores, test_scores, best_model = runTuneTest(clf, parameters, X, y)
    print("---------------")
    print("Random Forest")
    print("---------------" + "\n")
    for i in range(len(best_params)):
        print("Fold " + str(i+1) + ":")
        print(str(best_params[i]))
        print("Training Score: " + str(train_scores[i]) + "\n")
    print("Fold, Test Accuracy")
    for i in range(len(test_scores)):
        print(str(i+1) + ", " + str(test_scores[i]))
    print("\n")

    print("best model params:")
    print(str(best_model.get_params))

    return(best_model)

def run_support_vectors(X, y):
    """
    This function will take in data and run it on an SVM classifier.
    """
    clf = SVC()
    parameters = {"C": [1, 10, 100, 1000], "gamma": [0.0001 , 0.001, 0.01, 0.1, 1.0]}
    best_params, train_scores, test_scores, best_model = runTuneTest(clf, parameters, X, y)
    print("---------------")
    print("SVM")
    print("---------------" + "\n")
    for i in range(len(best_params)):
        print("Fold " + str(i+1) + ":")
        print(str(best_params[i]))
        print("Training Score: " + str(train_scores[i]) + "\n")
    print("Fold, Test Accuracy")
    for i in range(len(test_scores)):
        print(str(i+1) + ", " + str(test_scores[i]))
    print("\n")

    print("best model params:")
    print(str(best_model.get_params))

    return(best_model)

def confusion_matrix(model, X, y, datatype = "binary"):
    """
    This function takes in the test dataset and a model and creates a confusion
    matrix.
    """
    if datatype == "binary":
        matrix = np.zeros((2, 2))
        #X = data['data']
        #y = data['target']
        index = 0
        correct = 0
        predictions = model.predict(X)
        for example in X:
            if predictions[index] == 1:
                pred = 1
            if predictions[index] == -1:
                pred = 0
            if y[index] == 1:
                true = 1
            if y[index] == -1:
                true = 0
            if predictions[index] == y[index]:
                correct += 1
            matrix[true][pred] += 1
            index += 1
        accuracy = correct/index

    if datatype == "non-binary":
        matrix = np.zeros((8, 8))
        #X = data['data']
        #y = data['target']
        index = 0
        correct = 0
        predictions = model.predict(X)
        for example in X:
            if predictions[index] <= 4:
                pred = predictions[index] - 1
            if predictions[index] >= 7:
                pred = predictions[index] - 3
            if y[index] <= 4:
                true = y[index] - 1
            if y[index] >= 7:
                true = y[index] - 3
            if predictions[index] == y[index]:
                correct += 1
            #print(str(true))
            matrix[int(true)][int(pred)] += 1
            index += 1
        accuracy = correct/index

    return matrix, accuracy

def main():

    opts = util.parse_args()
    #data = util.data_preprocess()
    file = open('train_data.pkl', 'rb')
    test_file = open('test_data.pkl', 'rb')
    data = pickle.load(file)
    test_data = pickle.load(test_file)
    file.close()
    test_file.close()
    X = data['data']
    y = data['target']
    test_X = test_data['data']
    test_y = test_data['target']
    X,y = utils.shuffle(X,y) # shuffle the rows (utils is from sklearn)

    X = X[:2000] # only keep a certain number of examples
    y = y[:2000]

    test_X = test_X[:2000] # only keep a certain number of examples
    test_y = test_y[:2000]

    if opts.model == "KNN":
        best_model = run_KNN(X,y)

        print("")
        print("confusion matrix: \n")

        cma, accuracy = confusion_matrix(best_model.best_estimator_, test_X, test_y)

        print(str(cma))
        print("accuracy:")
        print(str(accuracy))

    elif opts.model == "RF":
        best_model = run_random_forest(X, y)

        print("")
        print("confusion matrix: \n")

        cma, accuracy = confusion_matrix(best_model.best_estimator_, test_X, test_y)

        print(str(cma))
        print("accuracy:")
        print(str(accuracy))

        print("feature importances: ") #print the most important features (words) for the random forest
        importances = best_model.best_estimator_.feature_importances_
        most_important = np.argsort(importances)
        most_important_shorter = most_important[:30]
        print(str(most_important_shorter))

        v = open("aclImdb/imdb.vocab", "r")
        vocab = v.readlines()
        v.close()

        print("highest-importance words printed first:")

        for i in most_important_shorter:
            print(vocab[i])

    elif opts.model == "SVM":
        best_model = run_support_vectors(X, y)

        print("")
        print("confusion matrix: \n")

        cma, accuracy = confusion_matrix(best_model.best_estimator_, test_X, test_y)

        print(str(cma))
        print("accuracy:")
        print(str(accuracy))

    else:
        print("Specify model to be run with -m")
        print("Options: KNN, RF, SVM")

if __name__ == "__main__":
    main()
