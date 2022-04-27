import random
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree, svm
from scipy.stats import multivariate_normal


def score(pred, true):
    conf = confusion_matrix(pred, true)
    acc = accuracy_score(pred, true)
    print(acc)
    print(conf)


def random_baseline(y):
    pred = [random.randint(0, 2) for x in range(len(y))]
    score(pred, y)
    
    
def decision_tree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    pred = clf.predict(X_test)
    score(pred, y_test)
    

def random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(10).fit(X_train, y_train)
    pred = clf.predict(X_test)
    score(pred, y_test)
    
    
def support_vector_machine(X_train, y_train, X_test, y_test):
    clf = svm.SVC().fit(X_train, y_train)
    pred = clf.predict(X_test)
    score(pred, y_test)
    
    
def mlp(X_train, y_train, X_test, y_test):
    clf = MLPClassifier().fit(X_train, y_train)
    pred = clf.predict(X_test)
    score(pred, y_test)


def multivariate_gauss(X, means, covs):
    class1 = multivariate_normal.pdf(X, means[0], covs[0], allow_singular=True)
    class2 = multivariate_normal.pdf(X, means[1], covs[1], allow_singular=True)
    class3 = multivariate_normal.pdf(X, means[2], covs[2], allow_singular=True)
    return np.argmax([class1, class2, class3])


def bayes_classification(X_train, y_train, X_test, y_test):
    means = []
    covs = []
    for i in set(y_train):
        df_class = X_train.loc[y_train == i]
        means.append(df_class.mean().tolist())
        covs.append(np.cov(df_class, rowvar=False))
    pred = X_test.apply(multivariate_gauss, axis=1, args=(means, covs))
    score(pred, y_test)
    
    
#TODO hw3 classifier
    
    
def parzen_window(X_train, y_train, X_test, y_test):
    pred = []
    for j, row in X_test.iterrows():
        ps = []
        for i in set(y_train):
            df_class = X_train.loc[y_train == i]
            p = 0
            for i, xi in df_class.iterrows():
                p += multivariate_normal.pdf(row-xi, [0,0], [[1,0],[0,1]])
            ps.append(p/len(df_class))
        pred.append(np.argmax(ps))
    score(pred, y_test)