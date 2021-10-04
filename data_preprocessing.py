import json
import numpy as np
import argparse
import torch
import sys
import os
import configparser

parser = configparser.ConfigParser()
parser.read("./configs/data_preprocessing.conf")

CHARACTER_BERT_PATH = parser.get("config", "characterBERT_path")
sys.path.append(CHARACTER_BERT_PATH)
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel

OUT_PATH = parser.get("config", "output_path")
ADE_INITIAL = parser.get("config", "ade_initial_dataset")


class Data_Proc:
    def __init__(self, initial_data_path, out_path, device):
        self.initial_data_path = initial_data_path
        self.out_path = out_path
        self.device = device
        self.ade_full = self.read_data()
        self.max_len = self.find_max_length()

    def read_data(self):
        # Read the initial ADE data
        with open(self.initial_data_path) as json_file:
            ade_full = json.load(json_file)

        return ade_full

    def find_max_length(self):
        # Find the maximum sequence length
        max_len = -1

        for s in self.ade_full:
            if len(s['tokens']) > max_len:
                max_len = len(s['tokens'])

        # Add 2 because we have '[CLS]' and '[SEP]' tokens as well
        max_len += 2

        return max_len

    def lower_casing(self, sent):
        lower_case_tokens = [t.lower() for t in sent['tokens']]
        return lower_case_tokens

    def get_ne_tags(self, tokens, sent):
        # Initialize a list with 'O' tags
        ne_tags = []
        for i in range(len(tokens)):
            ne_tags.append('O')

        # Update the list based on the entities
        for en in sent['entities']:
            if en['type'] == 'Adverse-Effect':
                ne_tags[en['start']] = 'B-AE'
                for i in range(en['start'] + 1, en['end']):
                    ne_tags[i] = 'I-AE'
            elif en['type'] == 'Drug':
                ne_tags[en['start']] = 'B-DRUG'
                for i in range(en['start'] + 1, en['end']):
                    ne_tags[i] = 'I-DRUG'

        return ne_tags

    def get_relation_pairs(self, sent):
        relation_pairs = []
        for r in sent['relations']:
            ae = sent['entities'][r['head']]
            drug = sent['entities'][r['tail']]
            relation_pairs.append([drug['end'] - 1, ae['end'] - 1])

        return relation_pairs

    def prepare_characterBert_input(self, tokens):
        # Add '[CLS]' and '[SEP]' tokens
        tokens = ['[CLS]', *tokens, '[SEP]']

        # Find how many 0s should be added
        to_be_padded = self.max_len - len(tokens)
        zero_sec = [0] * to_be_padded
        one_sec = [1] * len(tokens)

        # Padding
        padded_tokens = tokens + ['[PAD]'] * to_be_padded

        # Masking
        attention_masks = [one_sec + zero_sec]
        attention_masks = torch.LongTensor(attention_masks)

        # Convert token sequence into character indices
        indexer = CharacterIndexer()
        batch = [padded_tokens]  # This is a batch with a single token sequence
        batch_ids = indexer.as_padded_tensor(batch)

        return batch_ids, attention_masks

    def adj_matrix(self, tokens, pairs):
        # Initialize a matrix with 0s with dimensions equal to the length of tokens list.
        shape_adj = len(tokens)
        adj = np.zeros((shape_adj, shape_adj))
        # For all the relations (pairs)
        for p in pairs:
            adj[p[0]][p[1]] = 1
            adj[p[1]][p[0]] = 1

        # Convert numpy array to list
        adj_list = [list(r) for r in adj]

        # Normalize the matrix
        adj_matrix_norm = self.adj_matrix_normalization(np.matrix(adj_list))

        return adj_matrix_norm

    def adj_matrix_normalization(self, adj_matrix):
        # Normalization
        I = np.matrix(np.eye(len(adj_matrix)))
        # Add the identity matrix
        A_hat = adj_matrix + I
        # Find the degree array of A_hat
        D_hat = np.array(np.sum(A_hat, axis=0))[0]
        D_hat = np.matrix(np.diag(D_hat))
        # Normalize the matrix
        A_norm = D_hat ** -1 * A_hat

        return A_norm

    def save_json(self, token_list, ne_tags, relation_pairs, node_initialization, A, sent_ids, attention_mask, sent):
        dict_out = {'tokens': token_list,
                    'ne tags': ne_tags,
                    'relation pairs': relation_pairs,
                    'node initialization': node_initialization.tolist(),
                    'adjacency matrix - only last': A.tolist(),
                    'padded encoded sentence': sent_ids.tolist(),
                    'attention mask': attention_mask.tolist()}

        f_name = str(sent['orig_id']) + '.json'

        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        # Extract the last file
        with open(self.out_path + f_name, 'w') as fp:
            json.dump(dict_out, fp)

    def execute_preprocessing(self, characterBert_path):
        # Load the pre-trained CharacterBERT - medical version
        model = CharacterBertModel.from_pretrained(characterBert_path + '/pretrained-models/medical_character_bert/')

        counter = 0
        model.eval()
        for sent in self.ade_full:
            # Take a lower-cased version of the token list.
            token_list = self.lower_casing(sent)

            # Create the ne tag list.
            ne_tags = self.get_ne_tags(token_list, sent)

            # Create the relation-pair list.
            relation_pairs = self.get_relation_pairs(sent)

            # Create the padded encoded version of the sentence and the attention mask.
            sent_ids, attention_mask = self.prepare_characterBert_input(token_list)

            # Create the initial node embeddings by passing the sentence through
            # the pre-trained medical version of CharacterBERT (last layer output).
            sent_ids.to(self.device)
            attention_mask.to(self.device)

            # Take the embeddings.
            with torch.no_grad():
                # Feed batch to CharacterBERT & get the embeddings.
                embeddings_for_batch, _ = model(input_ids=sent_ids, attention_mask=attention_mask)

            # Initialize the node embeddings by selecting the corresponding embeddings from
            # the output of characterBERT, leaving out the '[CLS]', '[SEP]' and '[PAD]' tokens.
            node_initialization = embeddings_for_batch[:, 1:len(token_list) + 1, :]

            # Define the adjacency matrix based on the relation pairs.
            A = self.adj_matrix(token_list, relation_pairs)

            # Extracted the processed json file
            self.save_json(token_list, ne_tags, relation_pairs, node_initialization, A, sent_ids, attention_mask, sent)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed.' .format(counter))

if __name__ == "__main__":
    if not os.path.isdir(CHARACTER_BERT_PATH):
        print('The path of CharacterBERT model does not exist.')
        sys.exit()

    if not os.path.isfile(ADE_INITIAL):
        print('The path of initial ADE dataset does not exist.')
        sys.exit()

    # Define the running device by checking if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    process_data = Data_Proc(ADE_INITIAL, OUT_PATH, device)
    process_data.execute_preprocessing(CHARACTER_BERT_PATH)
