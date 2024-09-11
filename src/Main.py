from BreastCancerSet import BreastCancerSet
from IrisSet import IrisSet
from SoyBeanSet import SoyBeanSet
from HouseVoteSet import HouseVoteSet
from GlassSet import GlassSet
from NaiveBayes import NaiveBayes
from LossFunctions import precision, recall, accuracy
from HelperFunctions import test_cross
import numpy as np


# Breast Cancer Set
breast_cancer = BreastCancerSet()
naive_breast_cancer = NaiveBayes()
print("Breast Cancer Data")
data_folds, label_folds = naive_breast_cancer.cross_validate(breast_cancer, 10)
test_cross(data_folds, label_folds)

breast_cancer_noise = BreastCancerSet()
breast_cancer_noise.add_noise()
naive_breast_cancer_noise = NaiveBayes()

print("Breast cancer set with noise")
data_folds, label_folds = naive_breast_cancer_noise.cross_validate(breast_cancer_noise, 10)
test_cross(data_folds, label_folds)

# Iris Set
iris_set = IrisSet()
naive_iris = NaiveBayes()

print("Iris Data")
data_folds, label_folds = naive_iris.cross_validate(iris_set, 10)
test_cross(data_folds, label_folds)

# iris set with noise

iris_set_noise = IrisSet()
iris_set_noise.add_noise()
naive_iris_noise = NaiveBayes()

print("Iris Data With Noise")
data_folds, label_folds = naive_iris_noise.cross_validate(iris_set_noise, 10)
test_cross(data_folds, label_folds)

# House set
house_set = HouseVoteSet()
naive_house = NaiveBayes()

print("House Vote Data")
data_folds, label_folds = naive_house.cross_validate(house_set, 10)
test_cross(data_folds, label_folds)

# House set with noise
house_set_noise = HouseVoteSet()
house_set_noise.add_noise()
naive_house_noise = NaiveBayes()

print("House Vote Data With Noise")
data_folds, label_folds = naive_house_noise.cross_validate(house_set_noise, 10)
test_cross(data_folds, label_folds)

# Soy set

soy_set = SoyBeanSet()
naive_soy = NaiveBayes()

print("Soy Data")
data_folds, label_folds = naive_soy.cross_validate(soy_set, 10)
test_cross(data_folds, label_folds)

# Soy set with noise

soy_set_noise = SoyBeanSet()
soy_set_noise.add_noise()

naive_soy_noise = NaiveBayes()

print("Soy Data With Noise")
data_folds, label_folds = naive_soy_noise.cross_validate(soy_set_noise, 10)
test_cross(data_folds, label_folds)


# Glass Set
# when constructing input number of bins and number of classes to classify

for i in [8]:
    glass_set = GlassSet(i, 2)

    naive_glass = NaiveBayes()
    print("Glass Data with " + str(i) + " bins")
    data_folds, label_folds = naive_glass.cross_validate(glass_set, 10)
    test_cross(data_folds, label_folds)

    glass_set_noise = GlassSet(i, 2)
    glass_set.add_noise()

    naive_glass_noise = NaiveBayes()

    print("Glass Data With Noise and " + str(i) + " bins")
    data_folds, label_folds = naive_glass_noise.cross_validate(glass_set_noise, 10)
    test_cross(data_folds, label_folds)


