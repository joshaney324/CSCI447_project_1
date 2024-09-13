from BreastCancerSet import BreastCancerSet
from IrisSet import IrisSet
from SoyBeanSet import SoyBeanSet
from HouseVoteSet import HouseVoteSet
from GlassSet import GlassSet
from HelperFunctions import test_dataset
from VisuFunctions import plot_avgs


avgs_original_data = []
avgs_noisy_data = []

# Breast Cancer Set
print("Breast Cancer Set")
breast_cancer = BreastCancerSet()
ori_avgs, noise_avgs = test_dataset(breast_cancer)
avgs_original_data.append(ori_avgs)
avgs_noisy_data.append(noise_avgs)

# Iris Set
print("Iris Set")
iris_set = IrisSet()
ori_avgs, noise_avgs = test_dataset(iris_set)
avgs_original_data.append(ori_avgs)
avgs_noisy_data.append(noise_avgs)

# House set
print("House Set")
house_set = HouseVoteSet()
ori_avgs, noise_avgs = test_dataset(house_set)
avgs_original_data.append(ori_avgs)
avgs_noisy_data.append(noise_avgs)

# Soy set
print("Soy set")
soy_set = SoyBeanSet()
ori_avgs, noise_avgs = test_dataset(soy_set)
avgs_original_data.append(ori_avgs)
avgs_noisy_data.append(noise_avgs)


# Glass Set
# when constructing input number of bins and number of classes to classify
for i in [8]:
    print("Glass Set")
    glass_set = GlassSet(i, 7)
    ori_avgs, noise_avgs = test_dataset(glass_set)
    avgs_original_data.append(ori_avgs)
    avgs_noisy_data.append(noise_avgs)

# plot result averages
plot_avgs(avgs_original_data, "original_data_avgs")
plot_avgs(avgs_noisy_data, "noisy_data_avgs")