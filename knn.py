from data import *
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def run_knn(X,y,N, finalX, finalY):
    clf = KNeighborsClassifier(n_neighbors=N)
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
    N = 33
    train_sizes, train_scores, valid_scores, y_test, y_pred = run_knn(wine_X, wine_y, N, finalWineX, finalWineY)

    plt.figure()
    plt.title("K Nearest Neighbors Learning Curves: Wine Data, N = {}".format(N))
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


# def plot_flowers():
#     flower_X, flower_y = loadFlowerData()

#     train_sizes, train_scores, valid_scores, y_test, y_pred = run_knn(flower_X, flower_y, 3)

#     plt.figure()
#     plt.title("K Nearest Neighbors Learning Curves: Iris Data")
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(valid_scores, axis=1)
#     test_scores_std = np.std(valid_scores, axis=1)
#     plt.grid()
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion_matrix_wine = confusion_matrix(y_test, y_pred)
#     print("Accuracy: {}%".format(accuracy))
#     print("Confusion Matrix: \n", confusion_matrix_wine)
#     plt.show()

# def plot_heart():
#     heart_X, heart_y = loadHeart()

#     train_sizes, train_scores, valid_scores, y_test, y_pred = run_knn(heart_X, heart_y, 3)

#     plt.figure()
#     plt.title("K Nearest Neighbors Learning Curves: Heart Disease Data")
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(valid_scores, axis=1)
#     test_scores_std = np.std(valid_scores, axis=1)
#     plt.grid()
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion_matrix_wine = confusion_matrix(y_test, y_pred)
#     print("Accuracy: {}%".format(accuracy))
#     print("Confusion Matrix: \n", confusion_matrix_wine)
#     plt.show()



def plot_car():
    car_X, car_y, finalCarX, finalCarY = loadCar()
    N = 41
    train_sizes, train_scores, valid_scores, y_test, y_pred = run_knn(car_X, car_y, N, finalCarX, finalCarY)

    plt.figure()
    plt.title("K Nearest Neighbors Learning Curves: Car Data, N = {}".format(N))
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
    plot_wine()
    plot_car()


if __name__ == "__main__":
    main()