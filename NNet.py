# In the neural network terminology:
#
# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass.
# The higher the batch size, the more memory space you'll need.
# number of iterations = number of passes, each pass using [batch size] number of examples.
# To be clear, one pass = one forward pass + one backward pass
# (we do not count the forward pass and backward pass as two different passes).
# Example: if you have 1000 training examples, and your batch size is 500,
# then it will take 2 iterations to complete 1 epoch.


import numpy as np;
import matplotlib.pyplot as plt;
import mltools as ml;
from sklearn.neural_network import MLPClassifier;
from scipy import stats
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score;
from sklearn.model_selection import RandomizedSearchCV;
from sklearn.model_selection import GridSearchCV;
from sknn.mlp import Classifier, Layer
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

Xtr = np.loadtxt("data/X_train.txt")
Ytr = np.loadtxt("data/Y_train.txt")
# also load features of the test data (to be predicted)
Xte = np.loadtxt("data/X_test.txt")

scaler = StandardScaler()
scaler.fit(Xtr)
Xtr = Xtr[0:10000,:]
Ytr = Ytr[0:10000]
Xtr = scaler.transform(Xtr)
# not normalizing the data affects NN performance, hence scaling
Xte = scaler.transform(Xte)
#
# clf = MLPClassifier(activation='tanh', solver='adam', alpha=1e-5, hidden_layer_sizes=(5000, ), random_state=1)
#
# clf.fit(Xtr, Ytr);
# Yte = clf.predict_proba(Xte)[:,1]

#Cross-validation

#
def MLPwithRSCV(X_train, y_train, X_test, y_test):
    # hyper params
    param_grid = {
        # 'learning_rate': stats.uniform(0.01, 1.0),
        # 'hidden0__units': [x for x in range(30,80,10)],
        # 'hidden0__type': ["Sigmoid", "Tanh"],
        # 'hidden1__units': [x for x in range(30,80,10)]
        # 'hidden1__type': ["Sigmoid", "Tanh"]
        # 'learning_rule':[] ,

        # 'alpha': stats.uniform(0.001, 1.0),
        'activation': ["logistic", "tanh"],
        'solver': ["sgd"],
        'hidden_layer_sizes': [(x,) for x in range(10,70,10)]
    }

    mlp = MLPClassifier(learning_rate='adaptive', momentum=0.9, nesterovs_momentum=True, early_stopping=True)
    # mlp = Classifier(layers=[Layer("Tanh", units=50),Layer("Sigmoid", units=40), Layer("Sigmoid", units=35), Layer("Softmax")],learning_rate=1, n_iter=10,loss_type="mcc",regularize= "L2",verbose=None)
    # model
    rs = RandomizedSearchCV(mlp, param_distributions=param_grid, n_jobs=-1, verbose=10, cv=5)
    rs.fit(X_train, y_train)

    # rs = GridSearchCV(mlp, param_grid, n_jobs=-1, verbose=10)
    # rs.fit(X_train, y_train)

    # metric
    accu = rs.score(X_test, y_test) * 100
    print("Best accuracy: %.2f%%" % accu)
    print("Best params: %s" % rs.best_params_)

    return accu
# #
# # Run the randomized Search in order to find the best hyperparameters for the neural network
#
# Xtr, Xvi, Ytr, Yvi = train_test_split(Xtr, Ytr, test_size=0.1, random_state=0)
# MLPwithRSCV(Xtr, Ytr, Xvi, Yvi)

#Accuracy result
# Best accuracy: 69.27%
# Best params: {'alpha': 0.9017106541555798, 'activation': 'tanh', 'solver': 'lbfgs', 'hidden_layer_sizes': (50,)}

# Alpha = 0.01, hidden layer sizes = 100
# Score [ 0.69291535  0.68615     0.68765     0.68995     0.69035     0.6893
#   0.68935     0.69155     0.68905     0.68913446]
# Accuracy: 0.69 (+/- 0.00)

# Alpha = 0.01, hidden layer sizes = 200
# Score [ 0.69161542  0.6859      0.68675     0.68955     0.68965     0.69        0.6877
#   0.69115     0.6872      0.6900345 ]
# Accuracy: 0.69 (+/- 0.00)

# nn = Classifier(layers=[Layer("Tanh", units=50),Layer("Sigmoid", units=50), Layer("Softmax")], learning_rate=0.001, n_iter=25)
# nn.fit(Xtr, Ytr)
# clf = Classifier(layers=[Layer("Sigmoid",units=50),Layer("Sigmoid",units=50), Layer("Softmax")],learning_rate=0.15, n_iter=20,loss_type="mcc",regularize= "L2",verbose=None)
# scores = cross_val_score(clf, Xtr, Ytr, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
#
#
# #Ensembling NN
nBag = 25
m,n = Xtr.shape;
ensemble = np.empty(nBag,dtype=object)
clf = MLPClassifier(activation='tanh', solver='adam', alpha=1, hidden_layer_sizes=(50,))
# clf = Classifier(layers=[Layer("Tanh", units = 50), Layer("Sigmoid",units=40),Layer("Sigmoid",units=35), Layer("Softmax")],learning_rate=0.8, n_iter=10,loss_type="mcc",regularize= "L2",verbose=None)
for i in range(nBag):
    ind = np.floor(m * np.random.rand(m)).astype(int)  # Bootstrap sample a data set:
    Xi, Yi = Xtr[ind, :], Ytr[ind]  # select the data at those indices
    print("Fit ensemble of: " + str(i))
    ensemble[i] = clf.fit(Xi, Yi);  # Train a model on data Xi, Yi

mtest, ntest = Xte.shape
predictTest = np.zeros((mtest, 25));
for i in range(25):
    predictTest[:, i] = ensemble[i].predict_proba(Xte)[:,1]

predictTest = np.mean(predictTest,axis=1)
print(predictTest);

fh = open('predictions.csv','w') # open file for upload
fh.write('ID,Prob1\n') # output header line
for i,yi in enumerate(predictTest):
    fh.write('{},{}\n'.format(i,yi)) # output each prediction
fh.close() # close the file
