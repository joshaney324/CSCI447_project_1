from NaiveBayes import NaiveBayes
from LossFunctions import precision, recall, accuracy
import numpy as np


def test_cross(data_folds, label_folds):
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    num_folds = len(data_folds)

    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        naive = NaiveBayes()
        naive.set_probabilities(train_data, train_labels)
        predictions = naive.classify(test_data)

        precision_vals = np.array(precision(predictions, test_labels))
        recall_vals = np.array(recall(predictions, test_labels))
        accuracy_vals = np.array(accuracy(predictions, test_labels))

        precision_total = 0
        recall_total = 0
        accuracy_total = 0
        counter = 0
        for precision_val, recall_val, accuracy_val in zip(precision_vals, recall_vals, accuracy_vals):
            precision_total += float(precision_val[1])
            recall_total += float(recall_val[1])
            accuracy_total += float(accuracy_val[1])
            counter += 1

        precision_avg += precision_total / counter
        recall_avg += recall_total / counter
        accuracy_avg += accuracy_total / counter

        # print("Fold " + str(i + 1))
        # print("Precision: " + str(precision_total/counter))
        # print("Recall: " + str(recall_total/counter))
        # print("Accuracy: " + str(accuracy_total/counter))

    print("Average precision: " + str(precision_avg / num_folds))
    print("Average recall: " + str(recall_avg / num_folds))
    print("Average accuracy: " + str(accuracy_avg / num_folds))

