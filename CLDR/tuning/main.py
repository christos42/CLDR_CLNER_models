import json
import sys
import random
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import configparser
import math

from utils_ import *
from loss import *
from train import *


parser = configparser.ConfigParser()
parser.read("./../../configs/train_RE.conf")

CHARACTER_BERT_PATH = parser.get("config", "characterBERT_path")
sys.path.append(CHARACTER_BERT_PATH)
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel


# Read the given arguments
SPLIT_NUM = parser.get("config", "split_num")
EPOCHS = int(parser.get("config", "epochs"))
BATCH_SIZE = int(parser.get("config", "batch_size"))
NEG_SAMPLES = int(parser.get("config", "neg_samples"))
EARLY_STOPPING = int(parser.get("config", "early_stopping"))
PATH_OUT = parser.get("config", "path_out")
PATH_OUT = PATH_OUT + 'split_' + str(SPLIT_NUM) + '/'

# Create the output directory if it doesn't exist.
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)

# For the dataloader
ADJ_WEIGHT = float(parser.get("config", "adj_weight"))
PATH_IN_GENERAL = parser.get("config", "path_in_general")
PATH_IN_GRAPH = parser.get("config", "path_in_graph")


class GraphConvolution(nn.Module):
    # GCN layer based on https://arxiv.org/abs/1609.02907
    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize a matrix for the weights
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # If a bias vector will be included then initialize it.
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Initialization of the weights.
        if init == 'uniform':
            print('Uniform Initialization')
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print('Xavier Initialization')
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print('Kaiming Initialization')
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        # Implementation of Xavier Uniform
        nn.init.xavier_normal_(self.weight.data, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' ->' + str(self.out_features) + ')'


class Model_RE(nn.Module):
    def __init__(self, nfeat, nhid1, device, init_gc_weights, characterBERT_path):
        super(Model_RE, self).__init__()

        self.device = device

        # Graph part
        self.gc1 = GraphConvolution(nfeat, nhid1, init=init_gc_weights)
        self.activation_function = nn.ReLU()

        # Text encoder part
        self.characterBERT_path = characterBERT_path + 'pretrained-models/medical_character_bert/'
        self.characterBERT = CharacterBertModel.from_pretrained(self.characterBERT_path)

        # Freeze the first 6 encoding layers and the initial embedding layer
        modules = [self.characterBERT.embeddings, *self.characterBERT.encoder.layer[:6]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def graph_forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        x = self.gc1(x, adj)
        x = self.activation_function(x)
        x_conc = torch.flatten(x)
        x_conc_un = x_conc.unsqueeze(0)

        return x_conc_un

    def forward(self, x, adj, sent_id, mask, indexes_of_pairs):
        '''
            x: the feature matrix for the GCN part for the different graphs
            adj: the normalized adjacency matrix
            sent_id: the encoded sentence (containing [CLS], [SEP] and [PAD] tokens)
            mask: the masking of the sentence (indication of true or padded token)
        '''
        # Perform the forward pass on the GCN part
        graph_out = []
        for r_id in x:
            tmp_r = []
            for g in r_id:
                tmp_r.append(self.graph_forward(g, adj))

            # Create one tensor for the output of the GCN layer.
            # It has the concatenated output of each graph.
            tmp_r_tensor = torch.cat(tmp_r, 0)
            graph_out.append(tmp_r_tensor)

        # Perform the forward pass on the Text Encoder part
        # - 1: a tensor with the embeddings of the final layer for each token
        # - 2: a tensor with the average embeddings (all tokens considered) of the final layer
        output = self.characterBERT(sent_id, attention_mask=mask)

        # Isolate the representations which are related to the relations in the text.
        sent_out = []
        for r_id in indexes_of_pairs:
            tmp_r_sent = []
            for pair in r_id:
                selected_tokens = output[0][0][pair]
                # Create one representation for each relation using concatenation.
                relation_representation = torch.flatten(selected_tokens)
                relation_representation = self.activation_function(relation_representation)
                relation_representation_un = relation_representation.unsqueeze(0)
                tmp_r_sent.append(relation_representation_un)

            # Create one tensor for the selection of the tokens.
            # It has the concatenated "relation representations" based on encoder output tokens.
            # The first representation is the correct one and the rest are wrong (negative).
            tmp_r_sent_tensor = torch.cat(tmp_r_sent, 0)
            sent_out.append(tmp_r_sent_tensor)

        return graph_out, sent_out, output[0]


def prepare_dataloaders():
    # Read the file with the CV splits
    with open('../../cv_splits.json') as json_file:
        cv_splits = json.load(json_file)

    # Find the path files for training
    train_files = []
    for f in cv_splits['split_' + SPLIT_NUM]['train set']:
        #train_files.append('../../../Iteration_3/data/files/' + f + '.json')
        train_files.append(f + '.json')

    # Split the total training files into a training and development/validation set.
    # Define the validation set.
    # Set a random seed in order to take the same split.
    random.seed(42)
    indexes = list(np.arange(len(train_files)))
    # 10% of the set.
    val_samples_num = int(len(train_files) * 0.1)
    val_indexes = random.sample(indexes, val_samples_num)

    train_files_split = []
    val_files = []
    for i, f in enumerate(train_files):
        if i in val_indexes:
            val_files.append(f)
        else:
            train_files_split.append(f)

    # Define the data loaders
    dataset_train = ADE_Dataset(filenames=train_files_split,
                                path_in_general=PATH_IN_GENERAL,
                                path_in_graphs=PATH_IN_GRAPH,
                                number_of_negative_adj=NEG_SAMPLES,
                                adj_weight=ADJ_WEIGHT)
    dataset_val = ADE_Dataset(filenames=val_files,
                              path_in_general=PATH_IN_GENERAL,
                              path_in_graphs=PATH_IN_GRAPH,
                              number_of_negative_adj=NEG_SAMPLES,
                              adj_weight=ADJ_WEIGHT)

    # Dataloaders
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=my_collate)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=my_collate)

    return dataloader_train, dataloader_val


if __name__ == "__main__":
    # Define the running device by checking if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    model = Model_RE(nfeat=768,
                     nhid1=768,
                     device=device,
                     init_gc_weights='kaiming',
                     characterBERT_path = CHARACTER_BERT_PATH)

    model.to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)

    # Initialize the loss function
    loss_InfoNCE = InfoNCE_loss_vectorized(temperature=0.1)

    dataloader_train, dataloader_val = prepare_dataloaders()

    # Train the model
    model_trained, losses_dict = train(model = model,
                                       optimizer = optimizer,
                                       loss_graph_text = loss_InfoNCE,
                                       train_loader = dataloader_train,
                                       val_loader = dataloader_val,
                                       epochs = EPOCHS,
                                       early_stopping = EARLY_STOPPING,
                                       checkpoint_path = PATH_OUT,
                                       device = device)

    # Save the losses
    with open(PATH_OUT + 'losses.json', 'w') as fp:
        json.dump(losses_dict, fp)