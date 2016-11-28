# -*- coding: utf-8 -*-
# author: jun ma

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print(__doc__)

X = np.genfromtxt("data/X_train.txt", delimiter=' ')
Y = np.genfromtxt("data/Y_train.txt", delimiter=' ')

Xtr = X[0:10000]
Ytr = Y[0:10000]
Xte = X[10000:20000]
Yte = Y[10000:20000]
# Set the parameters by cross-validation

cost = [2 ** (-5), 2 ** (-3), 2 ** (-1), 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15]
g = [2 ** (-15), 2 ** (-13), 2 ** (-11), 2 ** (-9), 2 ** (-7), 2 ** (-5), 2 ** (-3), 2 ** (-1), 2 ** 1, 2 ** 3]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': g,
                     'C': cost},
                    {'kernel': ['linear'], 'C': cost}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1, cache_size=300), tuned_parameters, cv=5,
                       pre_dispatch='2*n_jobs',
                       n_jobs=1,
                       scoring='%s_macro' % score)
    X = preprocessing.scale(Xtr)

    clf.fit(X, Ytr)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    Xt = preprocessing.scale(Xte)
    y_true, y_pred = Yte, clf.predict(Xt)
    print(classification_report(y_true, y_pred))
    print()
