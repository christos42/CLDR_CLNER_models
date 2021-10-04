import seaborn as sns
import random
import os
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import configparser

parser = configparser.ConfigParser()
parser.read("./../configs/tSNE_analysis.conf")

SPLIT_NUM = int(parser.get("config", "split_num"))
PATH = './' + parser.get("config", "path")

class tSNE_analysis:
    def __init__(self, path, split_num):
        self.path = path
        self.split_num = split_num
        self.test_files = self.read_file_paths()
        self.embeddings, self.ne_tags = self.get_embeddings_tags()

    def read_file_paths(self):
        file_paths = []
        for root, dirs, files in os.walk(self.path + 'split_' + str(self.split_num) + '/test_set/'):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(self.path + 'split_' + str(self.split_num) + '/test_set/' + filename)

        return file_paths

    def get_embeddings_tags(self):
        embeddings = []
        ne_tags = []
        # for f in sampled_test_files:
        for f in self.test_files:
            with open(f) as json_file:
                data = json.load(json_file)

            embeddings.extend(data['embeddings - NER'])
            ne_tags.extend(data['ne tags'])

        return embeddings, ne_tags

    def create_plots(self):
        X_embedded = TSNE(n_components=2).fit_transform(self.embeddings)

        sns.set(rc={'figure.figsize': (11.7, 8.27)})

        palette = sns.color_palette("bright", 5)

        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=self.ne_tags, legend='full', palette=palette)

        if not os.path.exists('plots/split_' + str(self.split_num) + '/NER/'):
            os.makedirs('plots/split_' + str(self.split_num) + '/NER/')
        plt.savefig('plots/split_' + str(self.split_num) + '/NER/full_test_set.png')

        plt.close()

if __name__ == '__main__':
    tSNE_obj = tSNE_analysis(PATH, SPLIT_NUM)
    tSNE_obj.create_plots()