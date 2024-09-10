import numpy as np


def precision(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_precisions = []
    for class_instance in classes:
        true_positives = 0
        false_positives = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                true_positives += 1
            elif prediction == class_instance and prediction != label:
                false_positives += 1
        try:
            class_precisions.append(float(true_positives/(true_positives + false_positives)))
        except:
            class_precisions.append(0.0)
    output = list(zip(classes, class_precisions))
    return output


def recall(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_recalls = []
    for class_instance in classes:
        true_positives = 0
        false_negatives = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                true_positives += 1
            elif prediction != class_instance and class_instance == label:
                false_negatives += 1
        class_recalls.append(float(true_positives / (true_positives + false_negatives)))
    output = list(zip(classes, class_recalls))
    return output

#This function needs to be checked, but this implements accuracy as (TP+TN)/(TP+TN+FP+FN), rather than TP/(TP+TN+FP+FN)
def accuracy(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_accuracies = []
    for class_instance in classes:
        true = 0
        false = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance:
                if prediction == label:
                    true += 1
                else:
                    false += 1
            else:
                if prediction == label:
                    false += 1
                else:
                    true += 1
        class_accuracies.append(float(true / (true + false)))
    output = list(zip(classes, class_accuracies))
    return output

