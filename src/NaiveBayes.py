from collections import Counter
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = np.zeros((0, 0, 0))
        self.classes = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0

    def set_class_probabilities(self, dataset):
        class_list = []

        for row in dataset:
            class_list.append(row[-1])

        self.classes = list(set(class_list))

        class_counts = Counter(class_list)
        total_items = len(class_list)

        for classType, count in class_counts.items():
            self.class_probabilities[classType] = count/total_items

    #This function needs to take in the dataset with the attribute values and class values as indices of the 3d array (i.e. all class values must be integers from 0 to n-1, where n is the number classes. All probabilities for class one will be stored in the first index of the 3d array, all probabilities for class two will be stored in the second index, and so on). That needs to be accomplished in preprocessing, but it is fairly easy.
    def set_feature_probabilities(self, dataset):
        num_features, num_entries = dataset.shape
        num_features -= 1
        class_list = np.unique(dataset[:, num_features])
        num_classes = class_list.size
        class_counts = np.array[num_classes]
        num_feature_values = max([len(np.unique(dataset[:, i])) for i in range(num_features)])
        self.feature_probabilities.reshape(num_classes, num_features, num_feature_values)
        self.feature_probabilities.fill(1)
        d=1

        for i in range(num_entries):
            for j in (class_list):
                if (dataset[i, num_features] == j):
                    class_counts[j] += 1
                    for k in range(num_features):
                        for l in range(num_feature_values):
                            if (dataset[i, k] == l):
                                self.feature_probabilities[j, k, l] += 1
                self.feature_probabilities[j] /= (class_counts[j]+d)