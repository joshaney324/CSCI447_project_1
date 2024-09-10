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

