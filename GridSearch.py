import numpy as np
# import svm as svm
# from libsvm.python.svmutil import *
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_val_score

cost = [2 ** 11, 2 ** 13, 2 ** 15]
g = [2 ** (-15), 2 ** (-13), 2 ** (-11), 2 ** (-9), 2 ** (-7), 2 ** (-5), 2 ** (-3), 2 ** (-1), 2 ** 1, 2 ** 3]

X = np.genfromtxt("data/X_train.txt", delimiter=' ')
Y = np.genfromtxt("data/Y_train.txt", delimiter=' ')
Xtr = X[0:20000]
Ytr = Y[0:20000]
# Xte = X[10000:20000]
# Yte = Y[10000:20000]

best = {}
best['acc'] = 0
best['g'] = 0.0
best['c'] = 0.0
for i in range(len(cost)):
    print "loop.....{}".format(i)
    for j in range(len(g)):
        clf = svm.SVC(C=cost[i], gamma=g[j], kernel='rbf')
        Xt = preprocessing.scale(Xtr)
        scores = cross_val_score(clf, Xt, Ytr, cv=5)
        print "Accuracy: {} +/- {}, C={}, G = {})".format(scores.mean(), scores.std() * 2, cost[i], g[j])
        # Xts = a.scale(Xte
        # Ypr = clf.predict(Xts)