import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report


def getData(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        data = []
        target = []
        for row in reader:
            parameters = [float(i) for i in row[:-1]]
            data.append(np.array(parameters))
            target.append(row[-1])
        return data, target


def split(data, target, training_rate):
    pack = list(zip(data, target))
    random.shuffle(pack)
    data, target = zip(*pack)
    i = int(training_rate * len(data))
    return np.array(data[:i]), target[:i], np.array(data[i:]), target[i:]


def get_confusion_matrix(target, predicted, labels):
    cm = confusion_matrix(target, predicted, labels = labels)
    mcm = multilabel_confusion_matrix(target, predicted, labels = labels)
    # print (cm)
    # print (mcm)
    print(classification_report(target, predicted, labels = labels, zero_division = 1))

    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    np.seterr(divide = 'ignore', invalid = 'ignore')
    n = tn + tp + fn + fp
