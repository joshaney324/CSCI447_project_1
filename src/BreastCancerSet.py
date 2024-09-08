import csv
import numpy as np


class BreastCancerSet:
    def __init__(self):
        with open("../data/breast-cancer-wisconsin.data", "r") as data_file:
            self.data = list(csv.reader(data_file, delimiter=','))

        # find invalid rows and delete them
        invalid_rows = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = int(self.data[i][j])
                except ValueError:
                    invalid_rows.append(i)

        for row in invalid_rows:
            del self.data[row]

        # Remove the ID column
        for row in self.data:
            del row[0]
        
        self.data = np.array(self.data)
        np.random.shuffle(self.data)

    def get_data(self):
        return self.data[:, :-1]

    def get_labels(self):
        return self.data[:, -1]


