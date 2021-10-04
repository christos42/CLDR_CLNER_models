import os
import json
import configparser
import numpy as np
from cuml.neighbors import KNeighborsClassifier as KNN

parser = configparser.ConfigParser()
parser.read("../configs/knn_classification_RE.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))
K = int(parser.get("config", "k"))
PATH = parser.get("config", "path")

class KNN_classification:
    def __init__(self, path_in_out, split_num, k):
        self.path_in_out = path_in_out
        self.split_num = split_num
        # Final run: Training + Development set combined
        self.train_set_files = self.read_file_paths('train')
        self.test_set_files = self.read_file_paths('test')
        self.k = k

    def read_file_paths(self, set_type):
        file_paths = []
        for root, dirs, files in os.walk(self.path_in_out + 'split_' + str(self.split_num) + '/' + set_type + '_set/'):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(self.path_in_out + 'split_' + str(self.split_num) + '/' + set_type + '_set/' + filename)

        return file_paths

    def execute_classification(self):
        embeddings_pairs_train = []
        labels_train = []
        for f in self.train_set_files:
            with open(f) as json_file:
                data = json.load(json_file)

            for i, en1 in enumerate(data['embeddings - RE']):
                for j, en2 in enumerate(data['embeddings - RE'][i + 1:]):
                    embeddings_pairs_train.append(np.array((en1 + en2)))
                    if [i, j + i + 1] in data['relation pairs'] or [j + i + 1, i] in data['relation pairs']:
                        labels_train.append(1)
                    else:
                        labels_train.append(0)


        embeddings_pairs_train_arr = np.array(embeddings_pairs_train)
        labels_train_arr = np.array(labels_train)

        # Fit
        KNN_classifier = KNN(n_neighbors = self.k, algorithm = 'brute')
        KNN_classifier.fit(embeddings_pairs_train_arr, labels_train_arr)

        # Free some memory
        del embeddings_pairs_train_arr
        del labels_train_arr

        # Inference step
        print('Inference step')
        for count, f in enumerate(self.test_set_files):
            with open(f) as json_file:
                data = json.load(json_file)

            predicted_relations = []
            for i, en1 in enumerate(data['embeddings - RE']):
                for j, en2 in enumerate(data['embeddings - RE'][i + 1:]):
                    conc_candidate = np.array((en1 + en2))
                    conc_candidate = np.expand_dims(conc_candidate, axis=0)
                    prediction = KNN_classifier.predict(conc_candidate)
                    if prediction == 1:
                        predicted_relations.append([i, j + i + 1])

            # Update the dictionary with the results.
            f_name = f.split('/')[-1]
            with open('./results/split_' + str(self.split_num) + '/' + f_name) as json_file_classified:
                data_classified = json.load(json_file_classified)

            data_classified['predictions - RE'] = predicted_relations

            # Extract the updated file
            with open('./results/split_' + str(self.split_num) + '/' + f_name, 'w') as fp:
                json.dump(data_classified, fp)

            if count % 10 == 0:
                print('{} files completed'.format(count))


if __name__ == '__main__':
    KNN_obj = KNN_classification(PATH, SPLIT_NUM, K)
    KNN_obj.execute_classification()
