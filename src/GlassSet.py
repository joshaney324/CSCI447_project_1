import csv
import numpy as np
import random as random


class GlassSet:
    def __init__(self):
        with open("../data/glass.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))
        invalid_rows = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    invalid_rows.append(i)
        self.data = np.array(self.data[:-1])
        self.data = np.delete(self.data, 0, 1)
        for i in range(len(self.data[0]) - 1):
            column = self.data[:,i]
            bins = np.histogram_bin_edges(column, bins = "auto")

            bin_assignments = np.digitize(column, bins)
            for j in range(len(column)):
                self.data[j][i] = bin_assignments[j]
        np.random.shuffle(self.data)

    def get_data(self):
        return self.data[:, :-1]

    def get_labels(self):
        return self.data[:, -1]

    def add_noise(self):
        samples, features = np.shape(self.data[:, :-1])
        num_shuffled_features = int(features * .1 + 1)
        shuffled_cols = []
        curr_col = random.randint(0, features - 1)
        for i in range(num_shuffled_features):
            while curr_col in shuffled_cols:
                curr_col = random.randint(0, features - 1)

            feature_col = np.array(self.data[:, curr_col])
            np.random.shuffle(feature_col)

            self.data[:, curr_col] = feature_col
