from BreastCancerSet import BreastCancerSet
from IrisSet import IrisSet
from SoyBeanSet import SoyBeanSet
from HouseVoteSet import HouseVoteSet
from GlassSet import GlassSet
from NaiveBayes import NaiveBayes
import numpy as np

# Breast Cancer Set

breast_cancer = BreastCancerSet()
breast_cancer_data = breast_cancer.get_data()
breast_cancer_train = breast_cancer_data[:int(len(breast_cancer_data) * .6)]
breast_cancer_test = breast_cancer_data[:int(len(breast_cancer_data) * -.4):]
breast_cancer_labels = breast_cancer.get_labels()
breast_cancer_train_labels = breast_cancer_labels[:int(len(breast_cancer_data) * .6)]
breast_cancer_test_labels = breast_cancer_labels[:int(len(breast_cancer_data) * -.4):]

naive_breast_cancer = NaiveBayes()
naive_breast_cancer.set_probabilities(breast_cancer_train, breast_cancer_train_labels)
breast_cancer_predictions = naive_breast_cancer.classify(breast_cancer_test)

print("Breast Cancer Data")
print(np.mean(breast_cancer_predictions == breast_cancer_test_labels))
print(naive_breast_cancer.cross_validate(breast_cancer))

breast_cancer = BreastCancerSet()
breast_cancer.add_noise()
breast_cancer_data = breast_cancer.get_data()
breast_cancer_train = breast_cancer_data[:int(len(breast_cancer_data) * .6)]
breast_cancer_test = breast_cancer_data[:int(len(breast_cancer_data) * -.4):]
breast_cancer_labels = breast_cancer.get_labels()
breast_cancer_train_labels = breast_cancer_labels[:int(len(breast_cancer_data) * .6)]
breast_cancer_test_labels = breast_cancer_labels[:int(len(breast_cancer_data) * -.4):]

naive_breast_cancer = NaiveBayes()
naive_breast_cancer.set_probabilities(breast_cancer_train, breast_cancer_train_labels)
breast_cancer_predictions = naive_breast_cancer.classify(breast_cancer_test)

print("Breast cancer set with noise")
print(np.mean(breast_cancer_predictions == breast_cancer_test_labels))
print(naive_breast_cancer.cross_validate(breast_cancer))

# Iris Set

iris_set = IrisSet()
iris_data = iris_set.get_data()
iris_train = iris_data[:int(len(iris_data) * .6)]
iris_test = iris_data[:int(len(iris_data) * -.4):]
iris_labels = iris_set.get_labels()
iris_train_labels = iris_labels[:int(len(iris_data) * .6)]
iris_test_labels = iris_labels[:int(len(iris_data) * -.4):]

naive_iris = NaiveBayes()
naive_iris.set_probabilities(iris_train, iris_train_labels)
iris_predictions = naive_iris.classify(iris_test)

print("Iris Data")
print(np.mean(iris_predictions == iris_test_labels))
print(naive_iris.cross_validate(iris_set))


# iris set with noise

iris_set = IrisSet()
iris_set.add_noise()
iris_data = iris_set.get_data()
iris_train = iris_data[:int(len(iris_data) * .6)]
iris_test = iris_data[:int(len(iris_data) * -.4):]
iris_labels = iris_set.get_labels()
iris_train_labels = iris_labels[:int(len(iris_data) * .6)]
iris_test_labels = iris_labels[:int(len(iris_data) * -.4):]

naive_iris = NaiveBayes()
naive_iris.set_probabilities(iris_train, iris_train_labels)
iris_predictions = naive_iris.classify(iris_test)

print("Iris Data With Noise")
print(np.mean(iris_predictions == iris_test_labels))
print(naive_iris.cross_validate(iris_set))

# House set

house_set = HouseVoteSet()
house_data = house_set.get_data()
house_train = house_data[:int(len(house_data) * .6)]
house_test = house_data[int(len(house_data) * -.4):]
house_labels = house_set.get_labels()
house_train_labels = house_labels[:int(len(house_data) * .6)]
house_test_labels = house_labels[int(len(house_data) * -.4):]

naive_house = NaiveBayes()
naive_house.set_probabilities(house_train, house_train_labels)
house_predictions = naive_house.classify(house_test)

print("House Vote Data")
print(np.mean(house_predictions == house_test_labels))
print(naive_house.cross_validate(house_set))

# House set with noise
house_set = HouseVoteSet()
house_set.add_noise()
house_data = house_set.get_data()
house_train = house_data[:int(len(house_data) * .6)]
house_test = house_data[int(len(house_data) * -.4):]
house_labels = house_set.get_labels()
house_train_labels = house_labels[:int(len(house_data) * .6)]
house_test_labels = house_labels[int(len(house_data) * -.4):]

naive_house = NaiveBayes()
naive_house.set_probabilities(house_train, house_train_labels)
house_predictions = naive_house.classify(house_test)

print("House Vote Data With Noise")
print(np.mean(house_predictions == house_test_labels))
print(naive_house.cross_validate(house_set))

# Soy set

soy_set = SoyBeanSet()
soy_data = soy_set.get_data()
soy_train = soy_data[:int(len(soy_data) * .6)]
soy_test = soy_data[int(len(soy_data) * -.4):]
soy_labels = soy_set.get_labels()
soy_train_labels = soy_labels[:int(len(soy_data) * .6)]
soy_test_labels = soy_labels[int(len(soy_data) * -.4):]

naive_soy = NaiveBayes()
naive_soy.set_probabilities(soy_train, soy_train_labels)
soy_predictions = naive_soy.classify(soy_test)

print("Soy Data")
print(np.mean(soy_predictions == soy_test_labels))
print(naive_soy.cross_validate(soy_set))

# Soy set with noise

soy_set = SoyBeanSet()
soy_set.add_noise()
soy_data = soy_set.get_data()
soy_train = soy_data[:int(len(soy_data) * .6)]
soy_test = soy_data[int(len(soy_data) * -.4):]
soy_labels = soy_set.get_labels()
soy_train_labels = soy_labels[:int(len(soy_data) * .6)]
soy_test_labels = soy_labels[int(len(soy_data) * -.4):]

naive_soy = NaiveBayes()
naive_soy.set_probabilities(soy_train, soy_train_labels)
soy_predictions = naive_soy.classify(soy_test)

print("Soy Data With Noise")
print(np.mean(soy_predictions == soy_test_labels))
print(naive_soy.cross_validate(soy_set))

# Glass Set

for i in [3, 4, 5]:
    glass_set = GlassSet(i)
    glass_data = glass_set.get_data()
    glass_train = glass_data[:int(len(glass_data) * .6)]
    glass_test = glass_data[int(len(glass_data) * -.4):]
    glass_labels = glass_set.get_labels()
    glass_train_labels = glass_labels[:int(len(glass_data) * .6)]
    glass_test_labels = glass_labels[int(len(glass_data) * -.4):]

    naive_glass = NaiveBayes()
    naive_glass.set_probabilities(glass_train, glass_train_labels)
    glass_predictions = naive_glass.classify(glass_test)

    print("Glass Data with " + str(i) + " bins")
    print(np.mean(glass_predictions == glass_test_labels))
    print(naive_glass.cross_validate(glass_set))

    glass_set = GlassSet(i)
    glass_set.add_noise()
    glass_data = glass_set.get_data()
    glass_train = glass_data[:int(len(glass_data) * .6)]
    glass_test = glass_data[int(len(glass_data) * -.4):]
    glass_labels = glass_set.get_labels()
    glass_train_labels = glass_labels[:int(len(glass_data) * .6)]
    glass_test_labels = glass_labels[int(len(glass_data) * -.4):]

    naive_glass = NaiveBayes()
    naive_glass.set_probabilities(glass_train, glass_train_labels)
    glass_predictions = naive_glass.classify(glass_test)

    print("Glass Data With Noise and " + str(i) + " bins")
    print(np.mean(glass_predictions == glass_test_labels))
    print(naive_glass.cross_validate(glass_set))

