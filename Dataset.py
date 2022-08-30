import os
import pickle
import numpy as np

START_INDEX = 70
END_INDEX   = 75
PEOPLE_ID  = 14
VEHICLE_ID = 18

class Dataset:
    dir = "cifar-100-python/"
    # Fine classes of Superclass People
    def __init__(self, subclasses):
        if not os.path.exists(self.dir):
            raise Exception("CIFAR-100 dataset is not downloaded")

        self.subclasses = subclasses
        self.label = "fine_labels" if self.subclasses else "coarse_labels"

    def load_training_data(self):
        train_dict = self.unpickle(self.dir + "train")
        data = train_dict['data']
        target = np.array(train_dict[self.label])

        indexes = []
        # Select fine classes of People superclass
        if self.subclasses:
            for i in range(START_INDEX, END_INDEX):
                fine_index = np.argwhere(target == i)
                fine_index = fine_index.reshape((len(fine_index),))
                indexes.append(fine_index)
            indexes = np.concatenate(indexes)
        # Select People and Vehicle1 superclass
        else:
            people_indexes = np.argwhere(target == PEOPLE_ID)
            vehicle1_indexes = np.argwhere(target == VEHICLE_ID)
            # Reshape from 2D to 1D np array for data retrieval and concatenate
            people_indexes = people_indexes.reshape((len(people_indexes),))
            vehicle1_indexes = vehicle1_indexes.reshape((len(vehicle1_indexes),))
            indexes = np.concatenate((people_indexes, vehicle1_indexes))

        # Get data
        train_data = data[indexes]
        train_target = target[indexes]

        return train_data, train_target

    def load_testing_data(self):
        test_dict = self.unpickle(self.dir + "test")
        data = test_dict['data']
        target = np.array(test_dict[self.label])

        indexes = []
        # Select fine classes of People superclass
        if self.subclasses:
            for i in range(START_INDEX, END_INDEX):
                fine_index = np.argwhere(target == i)
                fine_index = fine_index.reshape((len(fine_index),))
                indexes.append(fine_index)
            indexes = np.concatenate(indexes)
        # Select People and Vehicle1 superclass
        else:
            people_indexes = np.argwhere(target == PEOPLE_ID)
            vehicle1_indexes = np.argwhere(target == VEHICLE_ID)
            # Reshape from 2D to 1D np array for data retrieval and concatenate
            people_indexes = people_indexes.reshape((len(people_indexes),))
            vehicle1_indexes = vehicle1_indexes.reshape((len(vehicle1_indexes),))
            indexes = np.concatenate((people_indexes, vehicle1_indexes))
        # Get data
        test_data = data[indexes]
        test_target = target[indexes]

        return test_data, test_target

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
