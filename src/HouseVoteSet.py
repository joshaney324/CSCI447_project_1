import csv
import numpy as np
import random as random


class HouseVoteSet:
    def __init__(self):
        with open("../data/house-votes-84.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))
        self.data = np.array(self.data[:-1])
        np.random.shuffle(self.data)

    def get_data(self):
        return self.data[:, 1:]

    def get_labels(self):
        return self.data[:, 1]

    def add_noise(self):
        samples, features = np.shape(self.data[:, 1:])
        num_shuffled_features = int(features * .1 + 1)
        shuffled_cols = []
        curr_col = random.randint(0, features + 1)
        for i in range(num_shuffled_features):
            while curr_col in shuffled_cols:
                curr_col = random.randint(0, features - 1)

            feature_col = np.array(self.data[:, curr_col])
            np.random.shuffle(feature_col)

            self.data[:, curr_col] = feature_col

