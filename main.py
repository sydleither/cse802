import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

from data import get_data, dimensionality_reduction, feature_selection
from bayes_classifiers import MultiVariateBayes, GaussianBayes, KNN


def random_baseline(y):
    pred = [random.randint(0, 2) for x in range(len(y))]
    score(pred, y)


def score(pred, true):
    conf = confusion_matrix(pred, true)
    acc = accuracy_score(pred, true)
    f1 = f1_score(pred, true, average=None)
    f1_macro = f1_score(pred, true, average='macro')
    print(f1)
    print(acc)
    print(conf)
    print(f1_macro)
    return conf, acc, f1, f1_macro
    
    
def predict(classifier, X_train, y_train, X_val, y_val, sfs_dir):
    if sfs_dir != None:
        sfs = feature_selection(X_train, y_train, classifier, n_features=5, direction=sfs_dir)
        X_train_reduce = X_train[X_train.columns[sfs.get_support()]]
        X_val_reduce = X_val[X_val.columns[sfs.get_support()]]
        print(X_val.columns[sfs.get_support()])
        clf = classifier.fit(X_train_reduce, y_train)
        pred = clf.predict(X_val_reduce)
    else:
        clf = classifier.fit(X_train, y_train)
        pred = clf.predict(X_val)
    return score(pred, y_val)


def main(test=False, z_score=False, pca=False, sfs_dir=None):
    df = get_data(z_score)
    label_map = dict([(y,x) for x,y in enumerate(sorted(set(df['readmitted'])))])
    y = [label_map[x] for x in df['readmitted']]
    X = df.drop('readmitted', axis=1)
    
    if pca:
        X = dimensionality_reduction(X)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=3)
    
    classifiers = [tree.DecisionTreeClassifier(), RandomForestClassifier(5), \
                   LinearSVC(), MLPClassifier(), MultiVariateBayes(), \
                   GaussianBayes(), KNN(5)]
    
    if not test:
        f1s = [[],[],[],[]]
        i = 0
        for train, val in StratifiedKFold(4).split(X_temp, y_temp): #60/20/20 split
            if pca:
                X_train, X_val = X_temp[train], X_temp[val]
            else:
                X_train, X_val = X_temp.iloc[train], X_temp.iloc[val]
            y_train, y_val = np.array(y_temp)[train], np.array(y_temp)[val]
    
            for classifier in classifiers:
                print(classifier)
                conf, acc, f1, f1_macro = predict(classifier, X_train, y_train, X_val, y_val, sfs_dir)
                f1s[i].append(f1_macro)
            i += 1
            
        print(' & '.join("%.2f"%e for e in np.mean(f1s, axis=0)))
        print(' & '.join("%.5f"%e for e in np.var(f1s, axis=0)))
        
    else:
        scores = [[],[],[]]
        for classifier in classifiers:
            print(classifier)
            conf, acc, f1, f1_macro = predict(classifier, X_temp, y_temp, X_test, y_test, sfs_dir)
    
            scores[0].append(acc)
            scores[1].append(conf)
            scores[2].append(f1_macro)
            
        print(' & '.join("%.2f"%e for e in scores[0]))
        g = ''
        for sc in scores[1]:
            x = '$\\begin{bmatrix} '
            for s in sc:
                x += ' & '.join(str(e) for e in s)
                x += ' \\\\ '
            x += '\\end{bmatrix}$'
            g += f'& {x} '
        print(g)
        print(' & '.join("%.2f"%e for e in scores[2]))


if __name__ == '__main__':
    main()