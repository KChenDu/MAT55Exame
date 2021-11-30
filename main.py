from utils import *
from classifier import Classifier


if __name__ == '__main__':
    data, target = getData('data/glass.csv')
    classes = list(dict.fromkeys(target))
    classifier = Classifier(classes, data[0].size)
    training_data, training_target, test_data, test_target = split(data, target, 0.9)
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)
