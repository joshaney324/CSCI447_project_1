import csv
import numpy as np

class BreastCancerSet:
    def __init__(self):
        with open("../data/breast-cancer-wisconsin.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = int(self.data[i][j])
                except ValueError:
                    self.data[i][j] = 0

        col1 = []
        for row in self.data:
            col1.append(row[0])

        num_bins = 3
        bins = np.histogram_bin_edges(col1, bins=num_bins)

        bin_indices = np.digitize(col1, bins)
        for i in range(len(self.data)):
            self.data[i][0] = int(bin_indices[i])



