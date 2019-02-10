from data import *
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split



def run_boosting(X,y, finalX, finalY):
    tree = DecisionTreeClassifier(min_samples_leaf=7, max_depth=4)
    clf = AdaBoostClassifier(base_estimator=tree, n_estimators=50)
    # Get learning curve data
    start = time() 
    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.array([0.55, 0.65, 0.75, 0.85, 0.95]))
    print("cross validation time: {}".format(time() - start))
    # Finally, train on cross validation data and test on withheld data
    clf.fit(X, y)
    y_pred = clf.predict(finalX)

    return train_sizes, train_scores, valid_scores, finalY, y_pred

def plot_wine():
    wine_X, wine_y, finalWineX, finalWineY = loadWineData()

    train_sizes, train_scores, valid_scores, y_test, y_pred = run_boosting(wine_X, wine_y, finalWineX, finalWineY)

    plt.figure()
    plt.title("Adaboost Classifier Learning Curves: Wine Data")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix_wine = confusion_matrix(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix: \n", confusion_matrix_wine)
    plt.show()


def plot_car():
    car_X, car_y, finalCarX, finalCarY = loadCar()

    train_sizes, train_scores, valid_scores, y_test, y_pred = run_boosting(car_X, car_y,finalCarX, finalCarY)

    plt.figure()
    plt.title("Adaboost Learning Curves: Car Data")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix_wine = confusion_matrix(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix: \n", confusion_matrix_wine)
    plt.show()

def main():
    #plot_wine()
    plot_car()


if __name__ == "__main__":
    main()