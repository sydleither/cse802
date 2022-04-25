import random
from sklearn.metrics import confusion_matrix, accuracy_score


def random_baseline(y):
    pred = [random.randint(0, 2) for x in range(len(y))]
    conf = confusion_matrix(pred, y)
    acc = accuracy_score(pred, y)
    print(acc)
    print(conf)