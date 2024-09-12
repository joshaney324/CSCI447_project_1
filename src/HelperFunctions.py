from NaiveBayes import NaiveBayes
from LossFunctions import precision, recall, accuracy
import matplotlib.pyplot as plt
import numpy as np


def cross_validate(data_folds, label_folds):
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds and a list of label folds. it does not
    # return anything but prints out the metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)

    # For each testing fold, set up a training and testing set and then append the loss function values
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

        # make all the data into np arrays so that naive bayes class can use them
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

        # get the averages of all the precision, recall, and accuracy values from all the folds
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

    print("Average precision: " + str(precision_avg / folds))
    print("Average recall: " + str(recall_avg / folds))
    print("Average accuracy: " + str(accuracy_avg / folds))

    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]

def get_folds(dataset, num_folds):

    # the get_folds function is meant to split the data up into a specified number of folds. this function takes in a
    # Dataset object as well as a specified number of folds. it then returns a list of all the data folds and label
    # folds

    labels = dataset.get_labels()
    data = dataset.get_data()

    # determine the number of instances of each class in each fold,
    # storing the values in a 2d numpy array (each row is a fold, each column is a class)
    classes, num_instances = np.unique(labels, return_counts=True)
    num_instances_perfold = np.zeros((num_folds, len(classes)), int)
    for i in range(len(num_instances_perfold[0])):
        for j in range(len(num_instances_perfold)):
            num_instances_perfold[j,i] = num_instances[i] // num_folds
        num_extra = num_instances[i] % num_folds
        for k in range(num_extra):
            num_instances_perfold[k,i] += 1

    # declare two lists of np arrays, each list entry representing a fold,
    # one list with data and one with labels
    label_folds = []
    for i in range(num_folds):
        label_folds.append(np.empty(shape=0))
    data_folds = []
    for i in range(num_folds):
        data_folds.append(np.empty(shape=(0, len(data[0]))))

    # iterate down the columns (classes) in the num_instances_perfold array,
    # then across the rows (folds) in the array,
    # then get the number of instances of that class in that fold,
    # then iterate through the labels to add them,
    # and remove the instances added to that fold from the data/labels classes to ensure uniqueness
    for i in range(len(num_instances_perfold[:,0])):
        for j in range(len(num_instances_perfold[i])):
            num_instances_infold = num_instances_perfold[i,j]
            k = 0
            while k < len(labels):
                if classes[j] == labels[k]:
                    label_folds[i] = np.append(label_folds[i], labels[k])
                    data_folds[i] = np.vstack((data_folds[i], data[k]))
                    data = np.delete(data, k, 0)
                    labels = np.delete(labels, k)
                    num_instances_infold -= 1
                    k -= 1
                if num_instances_infold == 0:
                    break
                k += 1
    # return a tuple of data_folds, label_folds
    return data_folds, label_folds

def plot_avgs(avg_data, filename):
    # round percentages
    for i in range(len(avg_data)):
        for j in range(len(avg_data[i])):
            avg_data[i][j] = round(avg_data[i][j]*100, 2)

    labels = ["Precision", "Recall", "Accuracy"]
    dataset_names = ["Breast Cancer Data", "Iris Data", "House Vote Data", "Soy Beans Data", "Glass Data"]
    x = np.arange(5)
    width = 0.75

    fig, ax = plt.subplots(layout='constrained')
    for i in range(3):
        avgs = [dataset[i] for dataset in avg_data]
        # offset to group bars at the same x-tick
        offset = (i - 1) * width
        rects = ax.barh(3*x + offset, avgs, width, label=labels[i])
        ax.set_yticks(3*x, labels=dataset_names)
        ax.bar_label(rects)

    ax.set(xlabel='percentage', title='Average results for precision, recall and accuracy', xlim=(0, 140))
    ax.set_xticks(np.arange(0, 101, step=20))  
    ax.legend(loc="upper right")

    plt.savefig("../output/" + filename + ".jpg")