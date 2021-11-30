import csv
import numpy as np
import random


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
