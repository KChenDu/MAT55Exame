from utils import *
from classifier import Classifier


if __name__ == '__main__':
    print("--------------------------------------------------------------------------------------")
    print("1. Using algorithms seen in class")
    print("--------------------------------------------------------------------------------------")
    data, target = getData('data/glass.csv')
    classes = list(dict.fromkeys(target))
    classifier = Classifier(classes, data[0].size)
    training_data, training_target, test_data, test_target = split(data, target, 0.9)
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)

    print("--------------------------------------------------------------------------------------")
    print("2. Using scipy least sqrt algorithm")
    print("--------------------------------------------------------------------------------------")
    classifier = Classifier(classes, data[0].size, "scipy_lstsq")
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)

    print("--------------------------------------------------------------------------------------")
    print("3. Similar to 1, but use scipy for QR decomposition")
    print("--------------------------------------------------------------------------------------")
    classifier = Classifier(classes, data[0].size, "scipy_qr")
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)
