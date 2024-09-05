from collections import Counter


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
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






