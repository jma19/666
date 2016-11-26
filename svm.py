# -*- coding: utf-8 -*-
# author: jun ma
from libsvm.python.svmutil import *
import numpy as np

class SvmClassifier(object):
    def __init__(self):
        """
        """

    def parse(self, X, Y):
        """
        parser array data into libsvm style
        """
        prob_y = []
        prob_x = []
        for i in range(len(X)):
            xi = {}
            for j in range(len(X[0])):
                xi[int(j + 1)] = float(X[i][j])
            prob_y += [float(Y[i])]
            prob_x += [xi]
        return (prob_x, prob_y)

    def train(self, X, Y, options=None):
        """
        -s svm_type : set type of SVM (default 0)-----C-SVC
        -t kernel_type : set type of kernel function (default 2)
        -g  gamma : set gamma in kernel function (default 1/num_features)
        -c penalty cost  default 1        """
        m = svm_train(Y, X, options);
        return m;

    def predict(self, Xte, Yte, m):
        p_label, p_acc, p_val = svm_predict(Yte, Xte, m)
        res = {}
        res['acc'] = p_acc[0]
        res["mse"] = p_acc[1]
        res["predict_res"] = p_label
        return res;

svm = SvmClassifier()
Y = [0, 1, 0, 1]
X = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 4, 5], [1, 2, 6, 5]]
prob_x, prob_y = svm.parse(X, Y)
m = svm.train(prob_x, prob_y, '-g 0.01 -c 0.1')
predict = svm.predict(prob_x, prob_y, m)

print predict



# print predict
# y, x = svm_read_problem('data/iris.scale.txt')
# m = svm_train(y[:140], x[:140], '-c 4')
# print x[140:]
# p_label, p_acc, p_val = svm_predict(y[140:], x[140:], m)
# print p_acc
