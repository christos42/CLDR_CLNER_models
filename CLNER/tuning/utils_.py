import json
import os
import torch
import shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader


def my_collate(batch):
    '''
        Custom collate function for the dataloader in
        order to handle the case where the adjacency
        matrices have different sizes (no padding).
    '''
    enc_sent = [item[0] for item in batch]
    atte_mask = [item[1] for item in batch]
    ne_tags = [item[2] for item in batch]

    return [enc_sent, atte_mask, ne_tags]


class ADE_Dataset(Dataset):
    def __init__(self, files, path_in_general):
        super().__init__()
        '''
            Args:
                files (list): list with paths of the json files
        '''
        self.files = files
        self.path_in_general = path_in_general

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load json file
        with open(self.path_in_general + self.files[idx]) as json_file:
            data = json.load(json_file)

        # Load the padded encoded sentence and the attention mask
        # which are the inputs of the text encoder.
        enc_sent = torch.LongTensor(data['padded encoded sentence'])
        atte_mask = torch.LongTensor(data['attention mask'])

        # Load the relation pairs, the NE tags and the word pieces
        # which will be used to apply CL in the sentence level.
        ne_tags = data['ne tags']

        return enc_sent, atte_mask, ne_tags