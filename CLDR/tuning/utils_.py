import json
import os
import torch
import shutil
import copy
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


def my_collate(batch):
    '''
        Custom collate function for the dataloader in
        order to handle the case where the adjacency
        matrices have different sizes (no padding).
    '''
    x = [item[0] for item in batch]
    adj = [item[1] for item in batch]
    enc_sent = [item[2] for item in batch]
    atte_mask = [item[3] for item in batch]
    token_indexes_shifted = [item[4] for item in batch]

    return [x, adj, enc_sent, atte_mask, token_indexes_shifted]


class ADE_Dataset(Dataset):
    def __init__(self, filenames, path_in_general, path_in_graphs, number_of_negative_adj, adj_weight=0.5):
        super().__init__()
        '''
            Args:
                files (list): list with paths of the json files
        '''
        self.filenames = filenames
        self.path_in_general = path_in_general
        self.path_in_graphs = path_in_graphs
        self.number_of_negative_adj_to_sample = number_of_negative_adj
        self.adj_weight = adj_weight

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load json file
        with open(self.path_in_general + self.filenames[idx]) as json_file:
            data_general = json.load(json_file)

        with open(self.path_in_graphs + self.filenames[idx]) as json_file:
            data_graphs = json.load(json_file)

        # Set up the node embeddings for each graph
        graphs_to_return = []
        token_indexes_pairs = []
        for k in data_graphs.keys():
            # If the key has indexes pairs continue, we take them into
            # account when we are loading node initializations.
            if k.split('_')[-1] in ['indexes', 'rev']:
                continue
            # Add the positive graph
            tmp_l = [data_graphs[k][0]]
            tmp_in = [data_graphs[k + '_indexes'][0]]
            # -1 because the first graph of the list is the positive one.
            number_of_negative_graphs = len(data_graphs[k]) - 1
            number_of_negative_graphs_rev = len(data_graphs[k + '_rev'])
            if self.number_of_negative_adj_to_sample == -1:
                # Take all the negative graphs
                tmp_l.extend(data_graphs[k][1:])
                tmp_l.extend(data_graphs[k + '_rev'])
                tmp_in.extend(data_graphs[k + '_indexes'][1:])
                tmp_in.extend(data_graphs[k + '_indexes_rev'])
            else:
                if number_of_negative_graphs > self.number_of_negative_adj_to_sample//2:
                    sampled_indexes = self.sampling_indexes(number_of_negative_graphs,
                                                            self.number_of_negative_adj_to_sample//2)
                    for s in sampled_indexes:
                        tmp_l.append(data_graphs[k][s])
                        tmp_in.append(data_graphs[k + '_indexes'][s])
                else:
                    tmp_l.extend(data_graphs[k][1:])
                    tmp_in.extend(data_graphs[k + '_indexes'][1:])

                # The reversed sampling
                if number_of_negative_graphs_rev > self.number_of_negative_adj_to_sample//2:
                    sampled_indexes = self.sampling_indexes(number_of_negative_graphs_rev,
                                                            self.number_of_negative_adj_to_sample//2, 1)
                    for s in sampled_indexes:
                        tmp_l.append(data_graphs[k + '_rev'][s])
                        tmp_in.append(data_graphs[k + '_indexes_rev'][s])
                else:
                    tmp_l.extend(data_graphs[k + '_rev'])
                    tmp_in.extend(data_graphs[k + '_indexes_rev'])

            graphs_to_return.append(tmp_l)
            token_indexes_pairs.append(tmp_in)

        # Convert the lists to tensors
        graphs_to_return_ready = []
        for r_graph in graphs_to_return:
            tmp_l = []
            for p in r_graph:
                tmp_l.append(torch.FloatTensor(p))

            graphs_to_return_ready.append(tmp_l)

        # Adjacency matrix: It is the same because we only have 2 nodes.
        # You can adjust the adjacency matrix based on the attention that you
        # want to pay to the self loop (diagonal elements in the matrix).
        adj_matrix = torch.FloatTensor(np.array([[self.adj_weight, 1 - self.adj_weight],
                                                 [1 - self.adj_weight, self.adj_weight]]))

        # Create also a list with the indexes of interest shifted by 1 because
        # in the text encoder we also have '[CLS]' token in the beginning.
        token_indexes_pairs_shifted = []
        for r in token_indexes_pairs:
            tmp_shift = []
            for p in r:
                tmp_shift.append([p[0] + 1, p[1] + 1])
            token_indexes_pairs_shifted.append(tmp_shift)

        # Load the padded encoded sentence and the attention mask
        # which are the inputs of the text encoder.
        enc_sent = torch.LongTensor(data_general['padded encoded sentence'])
        atte_mask = torch.LongTensor(data_general['attention mask'])

        # Return:
        # - graph_to_return_ready: the node embeddings for each graph
        # - adj_matrix: the adjacency matrix of the graphs
        # - enc_sent, atte_mask: the encoded sentence and the attention mask
        # - final_tokens: the tokens which are related to each graph
        # - final_indexes_of_interest: the indexes (based on the token list) which are related to each graph
        # - indexes_of_interest_shifted: pairs of indexes of the true relations in the text shifted by one
        #                                position because we have the '[CLS]' token in the beginning
        return graphs_to_return_ready, adj_matrix, enc_sent, atte_mask, token_indexes_pairs_shifted

    def sampling_indexes(self, number_of_negative_graphs, sampling_number, rev_id = 0):
        if rev_id == 0:
            possible_indexes = list(np.arange(1, number_of_negative_graphs + 1))
        else:
            possible_indexes = list(np.arange(0, number_of_negative_graphs))
        return random.sample(possible_indexes, sampling_number)