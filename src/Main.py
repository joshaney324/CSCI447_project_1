from BreastCancerSet import BreastCancerSet
from NaiveBayes import NaiveBayes
import numpy as np

breast_cancer = BreastCancerSet()
data = breast_cancer.get_data()
train = data[:int(len(data) * .6)]
test = data[-80:]
labels = breast_cancer.get_labels()
train_labels = labels[:int(len(data) * .6)]
test_labels = labels[-80:]
naive = NaiveBayes()
naive.set_probabilities(train, train_labels)
predictions = naive.classify(test)

print(np.mean(predictions == test_labels))

