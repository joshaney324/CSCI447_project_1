import numpy as np


def precision(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_precisions = []
    for class_instance in classes:
        tp = 0
        fp = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                tp += 1
            elif prediction == class_instance and prediction != label:
                fp += 1
        if tp + fp == 0:
            class_precisions.append(0)
        else:
            class_precisions.append(float(tp/(tp + fp)))

    return list(zip(classes, class_precisions))


def recall(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_recalls = []
    for class_instance in classes:
        tp = 0
        fn = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance and prediction == label:
                tp += 1
            elif prediction != class_instance and class_instance == label:
                fn += 1
        if tp + fn == 0:
            class_recalls.append(0)
        else:
            class_recalls.append(float(tp / (tp + fn)))
        
    return list(zip(classes, class_recalls))


# This function needs to be checked, but this implements accuracy as (TP+TN)/(TP+TN+FP+FN), rather than TP/(TP+TN+FP+FN)
def accuracy(predictions, labels):
    labels = np.array(labels)
    predictions = np.array(predictions)
    classes = np.unique(labels)
    class_accuracies = []
    for class_instance in classes:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for prediction, label in zip(predictions, labels):
            if prediction == class_instance:
                if prediction == label:
                    tp += 1
                else:
                    fp += 1
            else:
                if prediction == label:
                    tn += 1
                else:
                    fn += 1
        class_accuracies.append(float((tp + tn) / (tp + tn + fp + fn)))
        
    return list(zip(classes, class_accuracies))

