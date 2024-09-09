import numpy as np

# Use numpy for setting up arrays as well as getting unique values/counts


class NaiveBayes:

    # constructor
    def __init__(self):
        self.class_probabilities = {}
        self.attribute_probabilities = {}
        self.classes = None
        self.class_counts = {}
        self.d = 1

    def set_probabilities(self, data, labels):

        # This function is meant to set self.class_probabilities and self.attribute probabilities. This
        # function is the heart of the training method in the naive bayes algorithm. It takes in an n x m data array
        # and an n length array of labels. These arrays can be any type

        # get the number of samples and the number of features from the input array
        samples, features = data.shape
        self.d = features

        # Get unique classes and their counts
        self.classes, class_counts = np.unique(labels, return_counts=True)

        # create a tuple of classes connected with their counts using zip then loop and divide the count by #samples

        for class_instance, count in zip(self.classes, class_counts):
            self.class_probabilities[class_instance] = count / samples

        # if class dictionary does not exist create a dictionary
        for class_instance in self.classes:
            if class_instance not in self.attribute_probabilities:
                self.attribute_probabilities[class_instance] = {}
            if class_instance not in self.class_counts:
                self.class_counts[class_instance] = 0

            # Get examples of the current class
            class_data = data[labels == class_instance]
            self.class_counts[class_instance] = len(class_data)

            for feature_idx in range(features):
                # for each feature in the dataset, get every value that occurs for that feature as well as the counts
                # for that specific value
                # if dictionary key does not exist then create empy dictionary
                if feature_idx not in self.attribute_probabilities[class_instance]:
                    self.attribute_probabilities[class_instance][feature_idx] = {}

                feature_values, value_counts = np.unique(class_data[:, feature_idx], return_counts=True)
                # get total number of class instances and then add d
                total = len(class_data) + self.d
                # use zip function to combine all the unique values and their counts, then calculate the specific
                # attribute probability
                for value, count in zip(feature_values, value_counts):
                    if value not in self.attribute_probabilities[class_instance][feature_idx]:
                        self.attribute_probabilities[class_instance][feature_idx][value] = {}
                    self.attribute_probabilities[class_instance][feature_idx][value] = (count + 1) / total

    def calculate_total_probability(self, instance):
        total_probabilities = {}
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
                    self.attribute_probabilities[class_instance][feature_idx][attribute_value] = 1 / (self.class_counts[class_instance] + self.d)
                attribute_probability = self.attribute_probabilities[class_instance][feature_idx][attribute_value]
                specific_probabilities.append(attribute_probability)
            specific_probability = np.prod(specific_probabilities)
            # set the total probability as the class probability times the specific probability and return it
            total_probabilities[class_instance] = class_instance_probability * specific_probability
        return total_probabilities

    def classify(self, test_data):
        predictions = []
        # for each instance in the test data get the total probability of each class and then take the max to classify
        for instance in test_data:
            total_probabilities = self.calculate_total_probability(instance)
            predictions.append(max(total_probabilities, key=total_probabilities.get))
        return np.array(predictions)
    
    def cross_validate(self, dataset):
        labels = dataset.get_labels()
        data = dataset.get_data()
        samples = data.shape[0]
        folds = 10
        accuracy = 0
        fold_size = samples // folds
        for i in range(folds):
            if i != folds - 1:
                test_set = data[i*fold_size:(i+1)*fold_size]
                test_labels = labels[i*fold_size:(i+1)*fold_size]
                training_set = np.concatenate((data[:i*fold_size], data[(i+1)*fold_size:]))
                training_labels = np.concatenate((labels[:i*fold_size], labels[(i+1)*fold_size:]))
            else:
                test_set = data[i*fold_size:]
                test_labels = labels[i*fold_size:]
                training_set = data[:i*fold_size]
                training_labels = labels[:i*fold_size]
            self.set_probabilities(training_set, training_labels)
            predictions = self.classify(test_set)
            accuracy += np.mean(predictions == test_labels)
        accuracy /= folds
        return accuracy

            





# class NaiveBayes:
#     def __init__(self):
#         self.class_probabilities = {}
#         self.feature_probabilities = np.zeros((0, 0, 0))
#         self.classes = []
#         self.accuracy = 0
#         self.precision = 0
#         self.recall = 0
#
#     def set_class_probabilities(self, dataset):
#         class_list = []
#
#         for row in dataset:
#             class_list.append(row[-1])
#
#         self.classes = list(set(class_list))
#
#         class_counts = Counter(class_list)
#         total_items = len(class_list)
#
#         for classType, count in class_counts.items():
#             self.class_probabilities[classType] = count/total_items
#
#     #This function needs to take in the dataset with the attribute values and class values as indices of the 3d array (i.e. all class values must be integers from 0 to n-1, where n is the number classes. All probabilities for class one will be stored in the first index of the 3d array, all probabilities for class two will be stored in the second index, and so on). That needs to be accomplished in preprocessing, but it is fairly easy.
#     def set_feature_probabilities(self, dataset):
#
#         num_entries, num_features = dataset.shape
#         num_features -= 1  # Last column is the class label
#         class_list = np.unique(dataset[:, num_features])  # Unique class labels
#         num_classes = class_list.size
#         class_counts = np.zeros(num_classes, dtype=int)  # Initialize class counts
#
#         # Find the maximum number of unique values for any feature
#         num_feature_values = max([len(np.unique(dataset[:, i])) for i in range(num_features)])
#
#         # Initialize feature probabilities with ones (for smoothing)
#         self.feature_probabilities = np.ones((num_classes, num_features, num_feature_values))
#
#         smoothing_factor = 1  # Add-one smoothing
#
#         # Populate feature probabilities
#         for i in range(num_entries):
#             class_label = dataset[i, num_features]  # Get the class of the current row
#             class_index = np.where(class_list == class_label)[0][0]  # Index of the class
#
#             class_counts[class_index] += 1  # Increment the class count
#
#             for k in range(num_features):  # For each feature
#                 feature_value = dataset[i, k]
#                 self.feature_probabilities[class_index, k, feature_value] += 1  # Increment count for the feature
#
#         # Normalize the feature probabilities by dividing by class count + smoothing
#         for j in range(num_classes):
#             self.feature_probabilities[j] /= (class_counts[j] + smoothing_factor)
#
#     def classify(self, instance):
#         num_features = len(instance)
#         num_classes = self.feature_probabilities.shape[0]
#
#         probabilities = np.zeros(num_classes)
#
#         for class_idx in range(num_classes):
#             label = self.classes[class_idx]
#             probability = self.class_probabilities[label]
#
#             for k in range(num_features - 1):
#                 feature_value = instance[k]
#                 probability *= self.feature_probabilities[class_idx, feature_value]
#
#             probabilities[class_idx] = probability

        # return self.classes[np.argmax(probabilities)]

        #for i in range(num_entries):
            #for j in (class_list):
                #if (dataset[i, num_features] == j):
                    #class_counts[j] += 1
                    #for k in range(num_features):
                        #for l in range(num_feature_values):
                            #if (dataset[i, k] == l):
                                #self.feature_probabilities[j, k, l] += 1
                #self.feature_probabilities[j] /= (class_counts[j]+d)