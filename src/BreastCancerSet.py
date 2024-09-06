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
                    #or just delete that example from the dataset
                    #del self.data[i]

        #Remove the ID column
        for row in self.data:
            del row[0]
        
        self.data=np.array(self.data)


    def get_data(self):
        return self.data


