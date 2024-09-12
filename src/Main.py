from BreastCancerSet import BreastCancerSet
from IrisSet import IrisSet
from SoyBeanSet import SoyBeanSet
from HouseVoteSet import HouseVoteSet
from GlassSet import GlassSet
from NaiveBayes import NaiveBayes
from HelperFunctions import cross_validate, get_folds
from VisuFunctions import plot_avgs, plot_confusion_matrix

avgs_original_data = []
avgs_noisy_data = []

# Breast Cancer Set
breast_cancer = BreastCancerSet()
naive_breast_cancer = NaiveBayes()
print("Breast Cancer Data")
data_folds, label_folds = get_folds(breast_cancer, 10)
ori_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
#plot_confusion_matrix(matrix_total, "breast_cancer_matrix")

breast_cancer_noise = BreastCancerSet()
breast_cancer_noise.add_noise()
naive_breast_cancer_noise = NaiveBayes()

print("Breast cancer set with noise")
data_folds, label_folds = get_folds(breast_cancer_noise, 10)
noisy_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
#plot_confusion_matrix(matrix_total, "breast_cancer_matrix_noisy")

# Iris Set
iris_set = IrisSet()
naive_iris = NaiveBayes()

print("Iris Data")
data_folds, label_folds = get_folds(iris_set, 10)
ori_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
#plot_confusion_matrix(matrix_total, "iris_matrix")

# iris set with noise
iris_set_noise = IrisSet()
iris_set_noise.add_noise()
naive_iris_noise = NaiveBayes()

print("Iris Data With Noise")
data_folds, label_folds = get_folds(iris_set_noise, 10)
noisy_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
#plot_confusion_matrix(matrix_total, "iris_matrix_noisy")

# House set
house_set = HouseVoteSet()
naive_house = NaiveBayes()

print("House Vote Data")
data_folds, label_folds = get_folds(house_set, 10)
ori_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
#plot_confusion_matrix(matrix_total, "house_vote_matrix")

# House set with noise
house_set_noise = HouseVoteSet()
house_set_noise.add_noise()
naive_house_noise = NaiveBayes()

print("House Vote Data With Noise")
data_folds, label_folds = get_folds(house_set_noise, 10)
noisy_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
#plot_confusion_matrix(matrix_total, "house_vote_matrix_noisy")

# Soy set
soy_set = SoyBeanSet()
naive_soy = NaiveBayes()

print("Soy Data")
data_folds, label_folds = get_folds(soy_set, 10)
ori_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
#plot_confusion_matrix(matrix_total, "soy_set_matrix")

# Soy set with noise
soy_set_noise = SoyBeanSet()
soy_set_noise.add_noise()

naive_soy_noise = NaiveBayes()

print("Soy Data With Noise")
data_folds, label_folds = get_folds(soy_set_noise, 10)
noisy_avgs, matrix_total = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
#plot_confusion_matrix(matrix_total, "soy_set_matrix_noisy")

# Glass Set
# when constructing input number of bins and number of classes to classify
for i in [8]:
    glass_set = GlassSet(i, 2)

    naive_glass = NaiveBayes()
    print("Glass Data with " + str(i) + " bins")
    ori_avgs, matrix_total = cross_validate(data_folds, label_folds)
    avgs_original_data.append(ori_avgs)
    #plot_confusion_matrix(matrix_total, "glass_set_matrix")

    glass_set_noise = GlassSet(i, 2)
    glass_set.add_noise()

    naive_glass_noise = NaiveBayes()

    print("Glass Data With Noise and " + str(i) + " bins")
    data_folds, label_folds = get_folds(glass_set_noise, 10)
    noisy_avgs, matrix_total = cross_validate(data_folds, label_folds)
    avgs_noisy_data.append(noisy_avgs)
    #plot_confusion_matrix(matrix_total, "glass_set_matrix_noisy")

# plot result averages
plot_avgs(avgs_original_data, "avg_chart_original")
plot_avgs(avgs_noisy_data, "avg_chart_noisy")