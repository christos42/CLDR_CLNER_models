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
import math 
import configparser

from utils_ import *
from loss import *
from train import *


parser = configparser.ConfigParser()
parser.read("./../../configs/train_NER_final_run.conf")

CHARACTER_BERT_PATH = parser.get("config", "characterBERT_path")
sys.path.append(CHARACTER_BERT_PATH)
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel


# Read the given arguments
SPLIT_NUM = parser.get("config", "split_num")
EPOCHS = int(parser.get("config", "epochs"))
BATCH_SIZE = int(parser.get("config", "batch_size"))
SAMPLING_STRATEGY = int(parser.get("config", "sampling_strategy"))
NUMBER_OF_SAMPLES = int(parser.get("config", "number_of_samples"))
NUMBER_OF_POSITIVE_SAMPLES = int(parser.get("config", "number_of_positive_samples"))
PATH_OUT = parser.get("config", "path_out")
PATH_OUT = PATH_OUT + 'split_' + str(SPLIT_NUM) + '/'

# Create the output directory if it doesn't exist.
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)

# For the dataloader
PATH_IN_GENERAL = parser.get('config', 'path_in_general')


class Model_NE(nn.Module):
    def __init__(self, dropout_ne, device, characterBERT_path):
        super(Model_NE, self).__init__()

        self.device = device

        # Text encoder part
        self.characterBERT_path = characterBERT_path + '/pretrained-models/medical_character_bert/'
        self.characterBERT = CharacterBertModel.from_pretrained(self.characterBERT_path)

        # Freeze the first 6 encoding layers and the initial embedding layer
        modules = [self.characterBERT.embeddings, *self.characterBERT.encoder.layer[:6]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        # NE representation part
        self.fc1_ne = nn.Linear(768, 768)
        self.drop_ne_layer = nn.Dropout(dropout_ne)

    def forward(self, sent_id, mask):
        '''
            sent_id: the encoded sentence (containing [CLS], [SEP] and [PAD] tokens)
            mask: the masking of the sentence (indication of true or padded token)
        '''
        output = self.characterBERT(sent_id, attention_mask=mask)

        # Pass each token representation from a FC layer
        ne_rep = self.drop_ne_layer(output[0][:, 1:torch.nonzero(mask[0]).shape[0] - 1, :])
        ne_rep = self.fc1_ne(ne_rep)

        return ne_rep[0]


def prepare_dataloaders():
    # Read the file with the CV splits
    with open('../../cv_splits.json') as json_file:
        cv_splits = json.load(json_file)

    # Find the path files for training
    train_files = []
    for f in cv_splits['split_' + str(SPLIT_NUM)]['train set']:
        train_files.append(f + '.json')

    dataset_train = ADE_Dataset(files=train_files,
                                path_in_general=PATH_IN_GENERAL)

    # Dataloaders
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=my_collate)

    return dataloader_train


if __name__ == '__main__':
    # Define the running device by checking if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    model = Model_NE(dropout_ne=0,
                     device=device,
                     characterBERT_path = CHARACTER_BERT_PATH)

    model.to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)

    # Initialize the loss function
    loss_InfoNCE = InfoNCE_loss_ne_batch_level_alt_sampling(temperature = 0.1,
                                                            sampling_strategy = SAMPLING_STRATEGY,
                                                            number_of_samples = NUMBER_OF_SAMPLES,
                                                            number_of_positive_samples = NUMBER_OF_POSITIVE_SAMPLES)


    dataloader_train = prepare_dataloaders()

    # Train the model
    model_trained, losses_dict = train(model = model,
                                       optimizer = optimizer,
                                       loss = loss_InfoNCE,
                                       train_loader = dataloader_train,
                                       epochs = EPOCHS,
                                       checkpoint_path = PATH_OUT,
                                       device = device)

    # Save the losses
    with open(PATH_OUT + 'losses.json', 'w') as fp:
        json.dump(losses_dict, fp)