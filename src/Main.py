from BreastCancerSet import BreastCancerSet
from NaiveBayes import NaiveBayes
import numpy as np

breast_cancer = BreastCancerSet()
data = breast_cancer.get_data()
labels = breast_cancer.get_labels()
naive = NaiveBayes()
naive.set_probabilities(data, labels)
predictions = naive.classify(data)

print(np.mean(predictions == labels))

