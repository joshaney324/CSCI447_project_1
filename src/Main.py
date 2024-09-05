from BreastCancerSet import BreastCancerSet
from NaiveBayes import NaiveBayes

breast_cancer = BreastCancerSet()
data = breast_cancer.get_data()
naive = NaiveBayes()
naive.set_class_probabilities(data)