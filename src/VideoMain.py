from BreastCancerSet import BreastCancerSet
from HelperFunctions import get_folds
from NaiveBayes import NaiveBayes
import numpy as np
from LossFunctions import precision, recall, accuracy

breast_cancer_set = BreastCancerSet()
data_folds, label_folds = get_folds(breast_cancer_set, 10)

train_data = []
train_labels = []
test_data = []
test_labels = []

for j in range(len(data_folds)):
    if j != 1:
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


naive_bayes_model = NaiveBayes()
naive_bayes_model.set_probabilities(train_data, train_labels)

predictions = naive_bayes_model.classify(test_data)

precision_vals = np.array(precision(predictions, test_labels))
recall_vals = np.array(recall(predictions, test_labels))
accuracy_vals, matrix = accuracy(predictions, test_labels)
accuracy_vals = np.array(accuracy_vals)

precision_vals = precision_vals.astype(float)
recall_vals = recall_vals.astype(float)
accuracy_vals = accuracy_vals.astype(float)

print("Mean precision value of one fold:", np.mean(precision_vals[:, 1]))
print("Mean recall value of one fold:", np.mean(recall_vals[:, 1]))
print("Mean accuracy value of one fold:", np.mean(accuracy_vals[:, 1]))

