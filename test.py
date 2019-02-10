from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import learning_curve

iris = datasets.load_iris()

X = iris.data

y = iris.target



clf = SVC(kernel='linear')

#clf.fit(X,y)

data = learning_curve(clf, X,y)