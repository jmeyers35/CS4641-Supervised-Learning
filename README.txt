Datasets:
    My car dataset is included as car.csv. 
    The red wine dataset can be found at this link: 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

Running my code:
    The code for each individual algorithm is in a separate python script. All you need to do is open a terminal instance, cd into whatever directory the files are in, and run 'python <file>.py', where file is one of 'boosting', 'decision_tree', 'knn', 'neuralnet', or 'svm'.
    Make sure that car.csv is in this directory.
    The values for the hyperparameters of the classifiers may not be correct for both datasets, as I tested them separately but by default the script will use the same classifier for both.
    I describe the values of these hyperparameters in my analysis, but I've also reproduced them below for convenience.

    Decision Trees
    Wine: min_samples_leaf=7, max_depth=2
    Car: min_samples_leaf=7, max_depth=2

    Boosted Decision Tree
    Wine: max_depth of the DecisionTreeClassifier = 1
    Car: max_depth of the DecisionTreeClassifier = 4
    min_samples_leaf is 7 for both.

    KNN
    The hyperparameters are correct and separate in this file for both datasets.

    SVM
        Wine:
            linear: C = 1
            rbf: gamma=0.015, C=1
        Car:
            linear: C = 1
            rbf: gamma = 0.6, C = 5

    Neural Nets:
    Wine: hidden_layer_sizes=(6,), n_iterations=500
    Car: hidden_layer_sizes=(100,), n_iterations=500


    

    Running the scripts will run the cross validation, testing, and display the learning curves on the screen. The confusion matrix and accuracy score on the test data is printed to console. 
    The time in seconds it took to perform the cross-validation is also printed to console.

    The source code can be found at https://github.com/jmeyers35/CS4641-Supervised-Learning