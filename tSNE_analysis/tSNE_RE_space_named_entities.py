import seaborn as sns
import random
import os
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import configparser

parser = configparser.ConfigParser()
parser.read("./../configs/tSNE_analysis.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))

class tSNE_analysis:
    def __init__(self, split_num):
        self.split_num = split_num
        self.embeddings, self.labels, self.en1, self.en2 = self.read_data()
        self.embeddings = self.embeddings.tolist()
        self.labels = self.labels.tolist()
        self.indexes_1, self.correct_tokens = self.get_indexes_1()
        self.indexes_0_hard = self.get_hard_indexes_0()

    def read_data(self):
        # Read the file
        f = h5py.File('dataset/test_set_split_' + str(self.split_num) + '.hdf5', 'r')

        embeddings = f['embeddings']
        labels = f['labels']
        en1 = f['entity_1']
        en2 = f['entity_2']

        embeddings_ready = embeddings[:]
        labels_ready = labels[:]

        en1_ready = [str(el).strip('[]').strip('\'') for el in en1[:].astype(str)]
        en2_ready = [str(el).strip('[]').strip('\'') for el in en2[:].astype(str)]

        return embeddings_ready, labels_ready, en1_ready, en2_ready

    def get_indexes_1(self):
        indexes_1 = []
        indexes_0 = []
        correct_tokens = []
        for i, v in enumerate(self.labels):
            if v == 1:
                indexes_1.append(i)
                if self.en1[i] not in correct_tokens:
                    correct_tokens.append(self.en1[i])
                if self.en2[i] not in correct_tokens:
                    correct_tokens.append(self.en2[i])
            else:
                indexes_0.append(i)

        return indexes_1, correct_tokens

    def get_hard_indexes_0(self):
        indexes_0_hard = []
        for i, v in enumerate(self.labels):
            if v == 0:
                if self.en1[i] in self.correct_tokens or self.en2[i] in self.correct_tokens:
                    indexes_0_hard.append(i)

        return indexes_0_hard

    def get_sampled_indexes(self, num_hard):
        sampled_indexes = random.sample(self.indexes_0_hard, num_hard * len(self.indexes_1))
        sampled_indexes.extend(self.indexes_1)

        return sampled_indexes

    def get_NE_tags_token_repr(self):
        tokens_rep = []
        ne_tags = []
        for i in self.indexes_1:
            if self.embeddings[i][:768] not in tokens_rep:
                tokens_rep.append(self.embeddings[i][:768])
                ne_tags.append('Drug')
            if self.embeddings[i][768:] not in tokens_rep:
                tokens_rep.append(self.embeddings[i][768:])
                ne_tags.append('AE')

        return tokens_rep, ne_tags

    def create_plots(self):
        for n in [1, 5, 10, 20]:
            sampled_indexes = self.get_sampled_indexes(n)

            tokens_rep, ne_tags = self.get_NE_tags_token_repr()
            tokens_rep_rest = []
            ne_tags_rest = []
            for i in sampled_indexes:
                if self.embeddings[i][:768] not in tokens_rep:
                    tokens_rep_rest.append(self.embeddings[i][:768])
                    ne_tags_rest.append('Outside')
                if self.embeddings[i][768:] not in tokens_rep:
                    tokens_rep_rest.append(self.embeddings[i][768:])
                    ne_tags_rest.append('Outside')

            tokens_rep_all = tokens_rep + tokens_rep_rest
            ne_tags_all = ne_tags + ne_tags_rest

            X_embedded = TSNE(n_components=2).fit_transform(tokens_rep_all)

            sns.set(rc={'figure.figsize': (11.7, 8.27)})

            palette = sns.color_palette("bright", 3)

            sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=ne_tags_all, legend='full', palette=palette)

            if not os.path.exists('plots/split_' + str(self.split_num) + '/RE/'):
                os.makedirs('plots/split_' + str(self.split_num) + '/RE/')
            plt.savefig('plots/split_' + str(self.split_num) + '/RE/'+ 'hard_' + str(n * len(self.indexes_1)) + '_NE.png')

            plt.close()

if __name__ == '__main__':
    tSNE_obj = tSNE_analysis(SPLIT_NUM)
    tSNE_obj.create_plots()