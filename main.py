import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from data import get_data, dimensionality_reduction, feature_selection
from bayes_classifiers import MultiVariateBayes, GaussianBayes, ParzenWindow


def random_baseline(y):
    pred = [random.randint(0, 2) for x in range(len(y))]
    score(pred, y)


def score(pred, true):
    conf = confusion_matrix(pred, true)
    acc = accuracy_score(pred, true)
    f1 = f1_score(pred, true, average=None)
    print(f1)
    print(acc)
    print(conf)


def main(pca=False, sfs_dir=None):
    df = get_data()
    label_map = dict([(y,x) for x,y in enumerate(sorted(set(df['readmitted'])))])
    y = [label_map[x] for x in df['readmitted']]
    X = df.drop('readmitted', axis=1)
    
    if pca:
        X = dimensionality_reduction(X)

    #TODO save as seperate datasets when ready to test model parameters
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=3)
    
    classifiers = [tree.DecisionTreeClassifier(), RandomForestClassifier(10), \
                   LinearSVC(), MLPClassifier(), MultiVariateBayes(), \
                   GaussianBayes(), ParzenWindow()]
    for train, val in StratifiedKFold(4).split(X_temp, y_temp): #60/20/20 split
        if pca:
            X_train, X_val = X_temp[train], X_temp[val]
        else:
            X_train, X_val = X_temp.iloc[train], X_temp.iloc[val]
        y_train, y_val = np.array(y_temp)[train], np.array(y_temp)[val]

        for classifier in classifiers:
            print(classifier)
            
            if sfs_dir != None:
                sfs = feature_selection(X_train, y_train, classifier, n_features=None, direction=sfs_dir)
                X_train_reduce = X_train[X_train.columns[sfs.get_support()]]
                X_val_reduce = X_val[X_val.columns[sfs.get_support()]]
            
            clf = classifier.fit(X_train, y_train)
            pred = clf.predict(X_val)
            score(pred, y_val)

    
if __name__ == '__main__':
    main(pca=False, sfs_dir='forward')