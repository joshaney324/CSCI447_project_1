from BreastCancerSet import BreastCancerSet
from IrisSet import IrisSet
from SoyBeanSet import SoyBeanSet
from HouseVoteSet import HouseVoteSet
from GlassSet import GlassSet
from NaiveBayes import NaiveBayes
from HelperFunctions import cross_validate, get_folds
from VisuFunctions import plot_avgs, plot_boxplot
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

avgs_original_data = []
avgs_noisy_data = []

# Breast Cancer Set
breast_cancer = BreastCancerSet()
naive_breast_cancer = NaiveBayes()
print("Breast Cancer Data")
data_folds, label_folds = get_folds(breast_cancer, 10)
ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
predictions = np.array(predictions)
true_labels = np.array(true_labels)
avgs_original_data.append(ori_avgs)

cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()
# plot_confusion_matrix(matrix_total, "breast_cancer_dataset_matrix", ["Benign", "Malignant"])

breast_cancer_noise = BreastCancerSet()
breast_cancer_noise.add_noise()
naive_breast_cancer_noise = NaiveBayes()

print("Breast cancer set with noise")
data_folds, label_folds = get_folds(breast_cancer_noise, 10)
noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
# plot_confusion_matrix(matrix_total, "noisy_breast_cancer_dataset_matrix")

cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


plot_boxplot(ori_accuracies, noisy_accuracies, "breast_cancer_dataset_boxplot")

# Iris Set
iris_set = IrisSet()
naive_iris = NaiveBayes()

print("Iris Data")
data_folds, label_folds = get_folds(iris_set, 10)
ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
# plot_confusion_matrix(matrix_total, "iris_dataset_matrix")

cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


# iris set with noise
iris_set_noise = IrisSet()
iris_set_noise.add_noise()
naive_iris_noise = NaiveBayes()

print("Iris Data With Noise")
data_folds, label_folds = get_folds(iris_set_noise, 10)
noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
# plot_confusion_matrix(matrix_total, "noisy_iris_dataset_matrix")

cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


plot_boxplot(ori_accuracies, noisy_accuracies, "iris_dataset_boxplot")

# House set
house_set = HouseVoteSet()
naive_house = NaiveBayes()

print("House Vote Data")
data_folds, label_folds = get_folds(house_set, 10)
ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
# plot_confusion_matrix(matrix_total, "house_vote_dataset_matrix", ["Democrat", "Republican"])
cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

# House set with noise
house_set_noise = HouseVoteSet()
house_set_noise.add_noise()
naive_house_noise = NaiveBayes()

print("House Vote Data With Noise")
data_folds, label_folds = get_folds(house_set_noise, 10)
noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
# plot_confusion_matrix(matrix_total, "noisy_house_vote_dataset_matrix")
cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

plot_boxplot(ori_accuracies, noisy_accuracies, "house_vote_dataset_boxplot")

# Soy set
soy_set = SoyBeanSet()
naive_soy = NaiveBayes()

print("Soy Data")
data_folds, label_folds = get_folds(soy_set, 10)
ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_original_data.append(ori_avgs)
# plot_confusion_matrix(matrix_total, "soy_dataset_matrix")

# Soy set with noise
soy_set_noise = SoyBeanSet()
soy_set_noise.add_noise()
naive_soy_noise = NaiveBayes()

print("Soy Data With Noise")
data_folds, label_folds = get_folds(soy_set_noise, 10)
noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
avgs_noisy_data.append(noisy_avgs)
# plot_confusion_matrix(matrix_total, "noisy_soy_dataset_matrix")

cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

plot_boxplot(ori_accuracies, noisy_accuracies, "soy_dataset_boxplot")

# Glass Set
# when constructing input number of bins and number of classes to classify
for i in [8]:
    glass_set = GlassSet(i, 7)
    naive_glass = NaiveBayes()
    print("Glass Data with " + str(i) + " bins")
    ori_avgs, matrix_total, ori_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
    avgs_original_data.append(ori_avgs)
    # plot_confusion_matrix(matrix_total, "glass_dataset_matrix")
    cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels),
                yticklabels=np.unique(true_labels))

    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

    glass_set_noise = GlassSet(i, 7)
    glass_set.add_noise()
    naive_glass_noise = NaiveBayes()

    print("Glass Data With Noise and " + str(i) + " bins")
    data_folds, label_folds = get_folds(glass_set_noise, 10)
    noisy_avgs, matrix_total, noisy_accuracies, predictions, true_labels = cross_validate(data_folds, label_folds)
    avgs_noisy_data.append(noisy_avgs)
    # plot_confusion_matrix(matrix_total, "noisy_glass_dataset_matrix")
    cm = confusion_matrix(predictions, true_labels, labels=np.unique(true_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels),
                yticklabels=np.unique(true_labels))

    # Add labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

    plot_boxplot(ori_accuracies, noisy_accuracies, "glass_dataset_boxplot")

# plot result averages
plot_avgs(avgs_original_data, "original_data_avgs")
plot_avgs(avgs_noisy_data, "noisy_data_avgs")