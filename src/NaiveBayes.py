import numpy as np
# Use numpy for setting up arrays as well as getting unique values/counts


class NaiveBayes:

    # this class contains the naive bayes algorithm. it contains all the class probabilities and feature value
    # probabilities. it has functions that use this information to predict test examples from the data.

    def __init__(self):
        # constructor

        # class probabilities is a dictionary where the key is the class label and the value is the raw probability
        # for the class

        # attribute probabilities is a multi-layered dictionary where the key structure to get to the feature value
        # probability is class label, feature index, feature value. the value returned by this key is the probability
        # for this specific feature value

        # classes is a list of all the class labels in the data

        # class counts is a dictionary where the key is the class label and the value is the count of that class

        self.class_probabilities = {}
        self.attribute_probabilities = {}
        self.class_counts = {}
        self.classes = []

    def set_probabilities(self, data, labels):

        # the set_probabilities function is meant to set the class_probabilities, attribute_probabilities, class_counts,
        # and classes variables created in the constructor. it follows the instructions given in the project description
        # to set all of these probabilities. this function takes in a numpy array of data and then a numpy array of
        # labels. this function does not return anything

        # set up variables needed for calculation and get unique classes
        samples, features = np.shape(data)
        unique_classes = np.unique(labels)
        self.classes = unique_classes

        # loop through all the classes found in the data, set up dictionaries, and split data up for specific class
        for unique_class in unique_classes:

            # set up variables regarding the class data
            self.class_probabilities[unique_class] = {}
            self.attribute_probabilities[unique_class] = {}
            self.class_counts[unique_class] = {}
            unique_class_data = data[labels == unique_class]
            total_instances = len(unique_class_data)
            self.class_counts[unique_class] = len(unique_class_data)
            self.class_probabilities[unique_class] = total_instances/samples

            # loop through the feature indexes for the class, set up dictionaries, collect unique values and counts for
            # the feature
            for feature_idx in range(features):
                self.attribute_probabilities[unique_class][feature_idx] = {}
                unique_vals, counts = np.unique(unique_class_data[:, feature_idx], return_counts=True)

                # loop through the zipped list of unique values and counts, calculate the probability for the
                # feature value, and insert it into the dictionary
                for val, count in zip(unique_vals, counts):
                    self.attribute_probabilities[unique_class][feature_idx][val] = (count + 1) / (total_instances + features)

    def calculate_total_probability(self, instance):

        # the calculate_total_probability function is meant to calculate the probability of the class given the
        # features. this function takes in a numpy array of a single instance of data. it then returns a dictionary
        # where the key is a class and the value is the probability for that class

        total_probabilities = {}

        # loop through all the classes in the dataset and calculate the probabilities for the class
        for class_instance in self.classes:

            # get the specific probability for the class given
            class_instance_probability = self.class_probabilities[class_instance]
            specific_probabilities = []

            # for each feature in the instance get the attribute probabilities and append them to the specific
            # probability for this instance. Then set the specific probability
            # by multiplying all the specific probabilities
            for feature_idx in range(len(instance)):
                attribute_value = instance[feature_idx]

                # if attribute not found in training set calculate probability by just having 1
                if attribute_value not in self.attribute_probabilities[class_instance][feature_idx]:
                    self.attribute_probabilities[class_instance][feature_idx][attribute_value] = 1 / (self.class_counts[class_instance] + len(instance))

                # get the probability for the specific attribute value and add it to the specific probabilities list
                attribute_probability = self.attribute_probabilities[class_instance][feature_idx][attribute_value]
                specific_probabilities.append(attribute_probability)

            # set the total probability as the class probability times the product of the specific probabilities and
            # return it
            total_probabilities[class_instance] = class_instance_probability * np.prod(specific_probabilities)
        return total_probabilities

    def classify(self, test_data):

        # the classify function is meant to classify a set of data. it takes in a numpy array of data where each row
        # is an instance in the data. it then returns a numpy array of labels that it predicted

        predictions = []

        # for each instance in the test data get the total probability of each class and then take the max to classify
        for instance in test_data:
            total_probabilities = self.calculate_total_probability(instance)
            predictions.append(max(total_probabilities, key=total_probabilities.get))
        return np.array(predictions)

    def get_classes(self):
        return self.classes