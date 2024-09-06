from BreastCancerSet import BreastCancerSet
from NaiveBayes import NaiveBayes

# Test pushing to repository

breast_cancer = BreastCancerSet()
data = breast_cancer.get_data()
naive = NaiveBayes()
naive.set_class_probabilities(data)