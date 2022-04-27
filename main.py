from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

from data import get_data
from classify import random_baseline, decision_tree, random_forest, \
                     support_vector_machine, mlp, bayes_classification, \
                     parzen_window


def main():
    df = get_data()
    label_map = dict([(y,x) for x,y in enumerate(sorted(set(df['readmitted'])))])
    y = [label_map[x] for x in df['readmitted']]
    X = df.drop('readmitted', axis=1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    
    for train, val in StratifiedKFold(4).split(X_temp, y_temp): #60/20/20 split
        X_train, X_val = X_temp.iloc[train], X_temp.iloc[val]
        y_train, y_val = np.array(y_temp)[train], np.array(y_temp)[val]

        bayes_classification(X_train, y_train, X_val, y_val)

    
if __name__ == '__main__':
    main()