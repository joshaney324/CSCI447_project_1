import csv
import numpy as np


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

