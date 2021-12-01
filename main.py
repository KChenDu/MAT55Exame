from utils import *
from classifier import Classifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

'''
Accuracy: It gives you the overall accuracy of the model, meaning the fraction of the total samples that were correctly classified by the classifier. To calculate accuracy, use the following formula: (TP+TN)/(TP+TN+FP+FN).

Misclassification Rate: It tells you what fraction of predictions were incorrect. It is also known as Classification Error. You can calculate it using (FP+FN)/(TP+TN+FP+FN) or (1-Accuracy).

Precision: It tells you what fraction of predictions as a positive class were actually positive. To calculate precision, use the following formula: TP/(TP+FP).

Recall: It tells you what fraction of all positive samples were correctly predicted as positive by the classifier. It is also known as True Positive Rate (TPR), Sensitivity, Probability of Detection. To calculate Recall, use the following formula: TP/(TP+FN).

Specificity: It tells you what fraction of all negative samples are correctly predicted as negative by the classifier. It is also known as True Negative Rate (TNR). To calculate specificity, use the following formula: TN/(TN+FP).

F1-score: It combines precision and recall into a single measure. Mathematically it’s the harmonic mean of precision and recall. It can be calculated as follows:
    F1-score = 2 * (precision * recall) / (precision + recall)
'''

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

'''
    print("--------------------------------------------------------------------------------------")
    print("error rate")
    er = (fn + fp) / n
    print(er)
    print("--------------------------------------------------------------------------------------")
    print("true positive rate / recall")
    tpr = tp / (tp + fn)
    print(tpr)
    print("--------------------------------------------------------------------------------------")
    print("false positive rate")
    tpr = tp / (fp + tn)
    print(tpr)
    print("--------------------------------------------------------------------------------------")
    print("specifity")
    s = tn / (tn + fp)
    print(s)
    print("--------------------------------------------------------------------------------------")
    print("precision")
    s = tp / (tp + fp)
    print(s)
'''
            

if __name__ == '__main__':
    print("--------------------------------------------------------------------------------------")
    print("1. Using algorithms seen in class")
    print("--------------------------------------------------------------------------------------")
    data, target = getData('data/glass.csv')
    classes = list(dict.fromkeys(target))
    labels = [str(int) for int in classes] 
    classifier = Classifier(classes, data[0].size)
    training_data, training_target, test_data, test_target = split(data, target, 0.9)
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)
    get_confusion_matrix(test_target, predicted, labels)

    print("--------------------------------------------------------------------------------------")
    print("2. Using scipy least sqrt algorithm")
    print("--------------------------------------------------------------------------------------")
    classifier = Classifier(classes, data[0].size, "scipy_lstsq")
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)
    get_confusion_matrix(test_target, predicted, labels)

    print("--------------------------------------------------------------------------------------")
    print("3. Similar to 1, but use scipy for QR decomposition")
    print("--------------------------------------------------------------------------------------")
    classifier = Classifier(classes, data[0].size, "scipy_qr")
    classifier.train(training_data, training_target)
    predicted = classifier.evaluate(test_data)
    print(test_target)
    print(predicted)
    get_confusion_matrix(test_target, predicted, labels)
