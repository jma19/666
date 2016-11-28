from sklearn import svm, grid_search, datasets
from sklearn import preprocessing
import numpy as np
from time import clock

#
# from sklearn import preprocessing
# X = preprocessing.scale(X)



X = np.genfromtxt("data/X_train.txt", delimiter=' ')
Y = np.genfromtxt("data/Y_train.txt", delimiter=' ')

Xtr = X[0:2000]
Ytr = Y[0:2000]
Xte = X[3000:4000]
Yte = Y[3000:4000]

iris = datasets.load_iris()
# parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}


parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

start = clock()
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
X = preprocessing.scale(Xtr)

# clf.fit(iris.data, iris.target)
clf.fit(X, Ytr)
finish = clock()

print (finish - start) / 1000000

print(clf.best_params_)
