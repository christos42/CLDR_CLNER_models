import os
import json
import configparser
import numpy as np
from cuml.neighbors import KNeighborsClassifier as KNN

parser = configparser.ConfigParser()
parser.read("./../configs/knn_classification_NER.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))
K = int(parser.get("config", "k"))
PATH = parser.get("config", "path")

class KNN_classification:
    def __init__(self, path_in, split_num, k):
        self.path_in = path_in
        self.split_num = split_num
        self.train_set_files = self.read_file_paths('train')
        self.test_set_files = self.read_file_paths('test')
        self.k = k

        # Embeddings
        self.embeddings_train, self.labels_train = self.read_embeddings_labels()

    def read_file_paths(self, set_type):
        file_paths = []
        for root, dirs, files in os.walk(self.path_in + 'split_' + str(self.split_num) + '/' + set_type + '_set/'):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(self.path_in + 'split_' + str(self.split_num) + '/' + set_type + '_set/' + filename)

        return file_paths


    def read_embeddings_labels(self):
        mapping_ne_tags = {'B-DRUG': 0,
                           'I-DRUG': 1,
                           'B-AE': 2,
                           'I-AE': 3,
                           'O': 4}
        embeddings = []
        ne_tags = []
        for f in self.train_set_files:
            with open(f) as json_file:
                data = json.load(json_file)

            for emb in data['embeddings - NER']:
                embeddings.append(np.array(emb))
            for tag in data['ne tags']:
                ne_tags.append(mapping_ne_tags[tag])

        return np.array(embeddings), np.array(ne_tags)

    def execute_classification(self):
        # Fit
        KNN_classifier = KNN(n_neighbors = self.k, algorithm = 'brute')
        KNN_classifier.fit(self.embeddings_train, self.labels_train)

        # Take the NER predictions of the test set and update the files.
        mapping_ne_tags = {'0': 'B-DRUG',
                           '1': 'I-DRUG',
                           '2': 'B-AE',
                           '3': 'I-AE',
                           '4': 'O'}

        # Inference step
        print('Inference step')
        for count, f in enumerate(self.test_set_files):
            with open(f) as json_file:
                data = json.load(json_file)

            predictions = KNN_classifier.predict(np.array(data['embeddings - NER']))
            ne_tags_predicted = []
            for p in predictions:
                ne_tags_predicted.append(mapping_ne_tags[str(p)])

            updated_dict = {'tokens': data['tokens'],
                            'ne tags': data['ne tags'],
                            'relation pairs': data['relation pairs'],
                            'predictions - NER': ne_tags_predicted}

            f_name = f.split('/')[-1]

            # Create the output directory if it doesn't exist.
            if not os.path.exists('./results/split_' + str(self.split_num) + '/'):
                os.makedirs('./results/split_' + str(self.split_num) + '/')

            # Extract the updated file
            with open('./results/split_' + str(self.split_num) + '/' + f_name, 'w') as fp:
                json.dump(updated_dict, fp)

            '''
            # Update the dictionary with the results.
            f_name = f.split('/')[-1]
            with open('./results/split_' + str(self.split_num) + '/' + f_name) as json_file_classified:
                data_classified = json.load(json_file_classified)

            data_classified['predictions - NER'] = ne_tags_predicted

            # Extract the updated file
            with open('./results/split_' + str(self.split_num) + '/' + f_name, 'w') as fp:
                json.dump(data_classified, fp)
            '''

            if count % 50 == 0:
                print('{} files processed.' .format(count))


if __name__ == '__main__':
    KNN_obj = KNN_classification(PATH, SPLIT_NUM, K)
    KNN_obj.execute_classification()
    