import util
import numpy as np
import pickle
#import fc_nn

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
    best_model = learner

    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    for train_index, test_index in skf.split(X, y): # making each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GridSearchCV(learner, params, cv = 3, verbose = 10)
        clf.fit(X_train, y = y_train)
        best_params.append(clf.best_params_)
        train_scores.append(clf.score(X_train, y = y_train))
        new_score = clf.score(X_test, y = y_test)
        test_scores.append(new_score)
        if new_score >= test_score:
            test_score = new_score
            best_model = clf

    return best_params, train_scores, test_scores, best_model

def run_KNN(X,y):
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

    print("Testing best model...")
    modelscore = best_model.score(X, y)
    print(str(modelscore))
    print("params:")
    print(str(best_model.get_params))

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


def main():

    opts = util.parse_args()
    #data = util.data_preprocess()
    file = open('train_data.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    X = data['data']
    y = data['target']
    X,y = utils.shuffle(X,y) # shuffle the rows (utils is from sklearn)
    X = X[:2000] # only keep 1000 examples
    y = y[:2000]
    if opts.model == "KNN":
        run_KNN(X,y)
    elif opts.model == "RF":
        run_random_forest(X, y)
    elif opts.model == "SVM":
        run_support_vectors(X, y)
    else:
        print("Specify model to be run with -m")
        print("Options: KNN, RF, SVM")

if __name__ == "__main__":
    main()
