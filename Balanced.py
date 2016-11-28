import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


def err(Ytr, Ypr):
    y_t = np.array(Ytr)
    y_p = np.array(Ypr)
    return np.mean(Ypr.reshape(Ytr.shape) != Ytr)


X = np.genfromtxt("data/X_train.txt", delimiter=' ')
Y = np.genfromtxt("data/Y_train.txt", delimiter=' ')

Xtr = X[0:20000]
Ytr = Y[0:20000]
Xte = X[20000:40000]
Yte = Y[20000:40000]


clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(Xtr, Ytr)
Ypr = clf.predict(Xte)
print err(Ytr, Ypr)
