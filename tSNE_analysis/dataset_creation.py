import json
import numpy as np
import random
import os
import h5py
import sys
import configparser

parser = configparser.ConfigParser()
parser.read("./../configs/tSNE_analysis.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))
PATH = './' + parser.get("config", "path")

class Dataset:
    def __init__(self, path, split_num):
        self.path = path
        self.split_num = split_num
        self.test_files = self.read_file_paths()

    def read_file_paths(self):
        file_paths = []
        for root, dirs, files in os.walk(self.path + 'split_' + str(self.split_num) + '/test_set/'):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(self.path + 'split_' + str(self.split_num) + '/test_set/' + filename)

        return file_paths

    def create_h5py(self, embeddings, labels, entity_1, entity_2):
        # Convert lists to arrays
        embeddings_arr = np.array(embeddings)
        labels_arr = np.array(labels)

        entity_1_asciiList = [n.encode("ascii", "ignore") for n in entity_1]
        entity_2_asciiList = [n.encode("ascii", "ignore") for n in entity_2]

        # Create the file
        if not os.path.exists('dataset/'):
            os.makedirs('dataset/')

        f1 = h5py.File('dataset/test_set_split_' + str(self.split_num) + '.hdf5', 'w')

        # Write the embeddings and the labels
        dset1 = f1.create_dataset("embeddings", embeddings_arr.shape, dtype=np.float64, data=embeddings_arr)
        dset2 = f1.create_dataset("labels", labels_arr.shape, dtype=np.float64, data=labels_arr)
        dset3 = f1.create_dataset('entity_1', (len(entity_1_asciiList), 1), 'S40', entity_1_asciiList)
        dset4 = f1.create_dataset('entity_2', (len(entity_2_asciiList), 1), 'S40', entity_2_asciiList)

        # Close the file
        f1.close()

    def create_dataset(self):
        entity_1 = []
        entity_2 = []
        pair_representations = []
        labels = []
        for f in self.test_files:
            with open(f) as json_file:
                data = json.load(json_file)

            for i, en1 in enumerate(data['embeddings - RE']):
                for j, en2 in enumerate(data['embeddings - RE'][i + 1:]):
                    entity_1.append(data['tokens'][i])
                    entity_2.append(data['tokens'][j + i + 1])
                    pair_representations.append(en1 + en2)
                    if [i, j + i + 1] in data['relation pairs'] or [j + i + 1, i] in data['relation pairs']:
                        labels.append(1)
                    else:
                        labels.append(0)

        self. create_h5py(pair_representations,
                          labels,
                          entity_1,
                          entity_2)

if __name__ == '__main__':
    data_obj = Dataset(PATH, SPLIT_NUM)
    data_obj.create_dataset()