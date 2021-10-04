import torch
import torch.nn as nn
import numpy as np
from random import sample, shuffle


class InfoNCE_loss_ne_batch_level_alt_sampling(nn.Module):
    '''
        Contrastive Loss function based on:
        https://arxiv.org/abs/2010.00747
        (Contrastive Learning of Medical Visual Representations from Paired Images and Text)
    '''

    def __init__(self, temperature, sampling_strategy, number_of_samples, number_of_positive_samples):
        super(InfoNCE_loss_ne_batch_level_alt_sampling, self).__init__()
        self.temperature = temperature
        self.sampling_strategy = sampling_strategy
        self.number_of_samples = number_of_samples
        self.number_of_positive_samples = number_of_positive_samples

    def forward(self, representations, ne_tags):
        cos = nn.CosineSimilarity(dim=0)
        losses = []
        for i, rep1 in enumerate(representations):

            # Take the sampled indexes.
            if self.sampling_strategy == 1:
                sampled_indexes = self.sampling_indexes_1(ne_tags, i)
            elif self.sampling_strategy == 2:
                sampled_indexes = self.sampling_indexes_2(ne_tags, i)
            elif self.sampling_strategy == 3:
                sampled_indexes = self.sampling_indexes_3(ne_tags, representations, i)
            elif self.sampling_strategy == 4:
                sampled_indexes = self.sampling_indexes_4(ne_tags, representations, i)
            elif self.sampling_strategy == 5:
                sampled_indexes = self.sampling_indexes_5(ne_tags, i)
            else:
                print('No valid sampling strategy was given.')

            if self.sampling_strategy == 5:
                numerator = 0
                denominator = 0
                for j in sampled_indexes:
                    # If the two tokens have the same ne tag then update the numerator.
                    if ne_tags[i].split('-')[-1] == ne_tags[j].split('-')[-1]:
                        numerator += torch.exp(cos(rep1, representations[j]) / self.temperature)

                    # Update the denominator.
                    denominator += torch.exp(cos(rep1, representations[j]) / self.temperature)

                # Extreme case: If the batch has only one sample of a particular tag type then
                #               the numerator will be 0 and the loss will be infinite in order
                #               to avoid that just make a quick check of numerator's value.
                if numerator != 0:
                    losses.append(-torch.log(numerator / denominator))
                else:
                    pass
                    # print(ne_tags[i])
            else:
                numerator = 0
                denominator = 0
                for j in sampled_indexes:
                    # If the two tokens have the same ne tag then update the numerator.
                    if ne_tags[i] == ne_tags[j]:
                        numerator += torch.exp(cos(rep1, representations[j]) / self.temperature)

                    # Update the denominator.
                    denominator += torch.exp(cos(rep1, representations[j]) / self.temperature)

                # Extreme case: If the batch has only one sample of a particular tag type then
                #               the numerator will be 0 and the loss will be infinite in order
                #               to avoid that just make a quick check of numerator's value.
                if numerator != 0:
                    losses.append(-torch.log(numerator / denominator))
                else:
                    pass
                    #print(ne_tags[i])

        final_loss = 0
        for l in losses:
            final_loss += l

        final_loss *= (1 / len(losses))

        return final_loss

    def sampling_indexes_1(self, ne_tags, not_this_index):
        b_drug_indexes = []
        i_drug_indexes = []
        b_ae_indexes = []
        i_ae_indexes = []
        o_indexes = []

        sampled_indexes = []
        for i, t in enumerate(ne_tags):
            if i == not_this_index:
                continue
            if t == 'B-DRUG':
                b_drug_indexes.append(i)
            elif t == 'I-DRUG':
                i_drug_indexes.append(i)
            elif t == 'B-AE':
                b_ae_indexes.append(i)
            elif t == 'I-AE':
                i_ae_indexes.append(i)
            else:
                o_indexes.append(i)

        # Shuffle the lists.
        shuffle(b_drug_indexes)
        shuffle(i_drug_indexes)
        shuffle(b_ae_indexes)
        shuffle(i_ae_indexes)
        shuffle(o_indexes)

        # Add all the indexes that correspond to 'I-DRUG' tags.
        sampled_indexes.extend(i_drug_indexes)

        # Add 'B-DRUG' indexes
        if len(b_drug_indexes) >= 15:
            if len(sampled_indexes) + 15 < self.number_of_samples:
                sampled_indexes.extend(b_drug_indexes[:15])
            else:
                sampled_indexes.extend(b_drug_indexes[:(self.number_of_samples - len(sampled_indexes))])
        else:
            if len(sampled_indexes) + len(b_drug_indexes) < self.number_of_samples:
                sampled_indexes.extend(b_drug_indexes)
            else:
                sampled_indexes.extend(b_drug_indexes[:(self.number_of_samples - len(sampled_indexes))])

        # Add 'B-AE' indexes
        if len(b_ae_indexes) >= 15:
            if len(sampled_indexes) + 15 < self.number_of_samples:
                sampled_indexes.extend(b_ae_indexes[:15])
            else:
                sampled_indexes.extend(b_ae_indexes[:(self.number_of_samples - len(sampled_indexes))])
        else:
            if len(sampled_indexes) + len(b_ae_indexes) < self.number_of_samples:
                sampled_indexes.extend(b_ae_indexes)
            else:
                sampled_indexes.extend(b_ae_indexes[:(self.number_of_samples - len(sampled_indexes))])

        # Add 'I-AE' indexes
        if len(i_ae_indexes) >= 15:
            if len(sampled_indexes) + 15 < self.number_of_samples:
                sampled_indexes.extend(i_ae_indexes[:15])
            else:
                sampled_indexes.extend(i_ae_indexes[:(self.number_of_samples - len(sampled_indexes))])
        else:
            if len(sampled_indexes) + len(i_ae_indexes) < self.number_of_samples:
                sampled_indexes.extend(i_ae_indexes)
            else:
                sampled_indexes.extend(i_ae_indexes[:(self.number_of_samples - len(sampled_indexes))])

        # Add 'O' indexes
        if len(o_indexes) >= self.number_of_samples - len(sampled_indexes):
            sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
        else:
            sampled_indexes.extend(o_indexes)

        return sampled_indexes

    def sampling_indexes_2(self, ne_tags, not_this_index):
        b_drug_indexes = []
        i_drug_indexes = []
        b_ae_indexes = []
        i_ae_indexes = []
        o_indexes = []

        sampled_indexes = []
        for i, t in enumerate(ne_tags):
            if i == not_this_index:
                continue
            if t == 'B-DRUG':
                b_drug_indexes.append(i)
            elif t == 'I-DRUG':
                i_drug_indexes.append(i)
            elif t == 'B-AE':
                b_ae_indexes.append(i)
            elif t == 'I-AE':
                i_ae_indexes.append(i)
            else:
                o_indexes.append(i)

        # Shuffle the lists.
        shuffle(b_drug_indexes)
        shuffle(i_drug_indexes)
        shuffle(b_ae_indexes)
        shuffle(i_ae_indexes)
        shuffle(o_indexes)

        #########################################
        ######### POSITIVE SAMPLING #############
        #########################################
        # Sample positive pairs
        if ne_tags[not_this_index] == 'B-DRUG':
            if len(b_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_drug_indexes)
        elif ne_tags[not_this_index] == 'I-DRUG':
            if len(i_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_drug_indexes)
        elif ne_tags[not_this_index] == 'B-AE':
            if len(b_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_ae_indexes)
        elif ne_tags[not_this_index] == 'I-AE':
            if len(i_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_ae_indexes)
        else:
            if len(o_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(o_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(o_indexes)

        #########################################
        ######### NEGATIVE SAMPLING #############
        #########################################
        # Sample negative pairs
        # Find how many negative pairs you should sample
        number_of_negative_pairs = self.number_of_samples - len(sampled_indexes)

        ############
        # B - DRUG #
        ############
        if ne_tags[not_this_index] == 'B-DRUG':
            # 'I-DRUG' tags
            if len(i_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_drug_indexes)

            # 'B-AE' tags
            if len(b_ae_indexes) >= (number_of_negative_pairs//4):
                sampled_indexes.extend(b_ae_indexes[:(number_of_negative_pairs//4)])
            else:
                sampled_indexes.extend(b_ae_indexes)

            # 'I-AE' tags
            if len(i_ae_indexes) >= (number_of_negative_pairs//4):
                sampled_indexes.extend(i_ae_indexes[:(number_of_negative_pairs//4)])
            else:
                sampled_indexes.extend(i_ae_indexes)

            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)

        ############
        # I - DRUG #
        ############
        if ne_tags[not_this_index] == 'I-DRUG':
            # 'B-DRUG' tags
            if len(b_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_drug_indexes)

            # 'B-AE' tags
            if len(b_ae_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_ae_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_ae_indexes)

            # 'I-AE' tags
            if len(i_ae_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_ae_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_ae_indexes)

            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)

        ############
        ## B - AE ##
        ############
        if ne_tags[not_this_index] == 'B-AE':
            # 'B-DRUG' tags
            if len(b_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_drug_indexes)

            # 'I-DRUG' tags
            if len(i_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_drug_indexes)

            # 'I-AE' tags
            if len(i_ae_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_ae_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_ae_indexes)

            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)

        ############
        ## I - AE ##
        ############
        if ne_tags[not_this_index] == 'I-AE':
            # 'B-DRUG' tags
            if len(b_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_drug_indexes)

            # 'I-DRUG' tags
            if len(i_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_drug_indexes)

            # 'B-AE' tags
            if len(b_ae_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_ae_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_ae_indexes)

            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)


        ###########
        #### O ####
        ###########
        if ne_tags[not_this_index] == 'O':
            # 'B-DRUG' tags
            if len(b_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_drug_indexes)

            # 'I-DRUG' tags
            if len(i_drug_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(i_drug_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(i_drug_indexes)

            # 'B-AE' tags
            if len(b_ae_indexes) >= (number_of_negative_pairs // 4):
                sampled_indexes.extend(b_ae_indexes[:(number_of_negative_pairs // 4)])
            else:
                sampled_indexes.extend(b_ae_indexes)

            # 'I-AE' tags
            if len(i_ae_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(i_ae_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(i_ae_indexes)

        shuffle(sampled_indexes)
        return sampled_indexes

    def sampling_indexes_3(self, ne_tags, representations, not_this_index):
        b_drug_indexes = []
        i_drug_indexes = []
        b_ae_indexes = []
        i_ae_indexes = []
        o_indexes = []

        sampled_indexes = []
        for i, t in enumerate(ne_tags):
            if i == not_this_index:
                continue
            if t == 'B-DRUG':
                b_drug_indexes.append(i)
            elif t == 'I-DRUG':
                i_drug_indexes.append(i)
            elif t == 'B-AE':
                b_ae_indexes.append(i)
            elif t == 'I-AE':
                i_ae_indexes.append(i)
            else:
                o_indexes.append(i)

        # Shuffle the lists.
        shuffle(b_drug_indexes)
        shuffle(i_drug_indexes)
        shuffle(b_ae_indexes)
        shuffle(i_ae_indexes)
        shuffle(o_indexes)

        #########################################
        ######### POSITIVE SAMPLING #############
        #########################################
        # Sample positive pairs
        if ne_tags[not_this_index] == 'B-DRUG':
            if len(b_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_drug_indexes)
        elif ne_tags[not_this_index] == 'I-DRUG':
            if len(i_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_drug_indexes)
        elif ne_tags[not_this_index] == 'B-AE':
            if len(b_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_ae_indexes)
        elif ne_tags[not_this_index] == 'I-AE':
            if len(i_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_ae_indexes)
        else:
            if len(o_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(o_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(o_indexes)

        #########################################
        ######### NEGATIVE SAMPLING #############
        #########################################
        # Find how many negative pairs you should sample
        number_of_negative_samples = self.number_of_samples - len(sampled_indexes)
        # Sample 'hard' negative pairs based on the distance of the representations
        samples_negative = self.find_k_closest_representations_euclidean(representations, ne_tags,
                                                                         not_this_index, number_of_negative_samples)

        # Add the negative samples in the sampling set
        sampled_indexes = sampled_indexes + samples_negative

        shuffle(sampled_indexes)
        return sampled_indexes

    def sampling_indexes_4(self, ne_tags, representations, not_this_index):
        b_drug_indexes = []
        i_drug_indexes = []
        b_ae_indexes = []
        i_ae_indexes = []
        o_indexes = []

        sampled_indexes = []
        for i, t in enumerate(ne_tags):
            if i == not_this_index:
                continue
            if t == 'B-DRUG':
                b_drug_indexes.append(i)
            elif t == 'I-DRUG':
                i_drug_indexes.append(i)
            elif t == 'B-AE':
                b_ae_indexes.append(i)
            elif t == 'I-AE':
                i_ae_indexes.append(i)
            else:
                o_indexes.append(i)

        # Shuffle the lists.
        shuffle(b_drug_indexes)
        shuffle(i_drug_indexes)
        shuffle(b_ae_indexes)
        shuffle(i_ae_indexes)
        shuffle(o_indexes)

        #########################################
        ######### POSITIVE SAMPLING #############
        #########################################
        # Sample positive pairs
        if ne_tags[not_this_index] == 'B-DRUG':
            if len(b_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_drug_indexes)
        elif ne_tags[not_this_index] == 'I-DRUG':
            if len(i_drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_drug_indexes)
        elif ne_tags[not_this_index] == 'B-AE':
            if len(b_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(b_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(b_ae_indexes)
        elif ne_tags[not_this_index] == 'I-AE':
            if len(i_ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(i_ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(i_ae_indexes)
        else:
            if len(o_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(o_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(o_indexes)

        #########################################
        ######### NEGATIVE SAMPLING #############
        #########################################
        # Find how many negative pairs you should sample
        number_of_negative_samples = self.number_of_samples - len(sampled_indexes)
        # Sample 'hard' negative pairs based on the distance of the representations
        samples_negative = self.find_k_closest_representations_cosine(representations, ne_tags,
                                                                      not_this_index, number_of_negative_samples)

        # Add the negative samples in the sampling set
        sampled_indexes = sampled_indexes + samples_negative

        shuffle(sampled_indexes)
        return sampled_indexes

    def sampling_indexes_5(self, ne_tags, not_this_index):
        drug_indexes = []
        ae_indexes = []
        o_indexes = []

        sampled_indexes = []
        for i, t in enumerate(ne_tags):
            if i == not_this_index:
                continue
            if t == 'B-DRUG':
                drug_indexes.append(i)
            elif t == 'I-DRUG':
                drug_indexes.append(i)
            elif t == 'B-AE':
                ae_indexes.append(i)
            elif t == 'I-AE':
                ae_indexes.append(i)
            else:
                o_indexes.append(i)

        # Shuffle the lists.
        shuffle(drug_indexes)
        shuffle(ae_indexes)
        shuffle(o_indexes)

        #########################################
        ######### POSITIVE SAMPLING #############
        #########################################
        # Sample positive pairs
        if ne_tags[not_this_index] == 'B-DRUG' or ne_tags[not_this_index] == 'I-DRUG':
            if len(drug_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(drug_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(drug_indexes)
        elif ne_tags[not_this_index] == 'B-AE' or ne_tags[not_this_index] == 'I-AE':
            if len(ae_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(ae_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(ae_indexes)
        else:
            if len(o_indexes) >= self.number_of_positive_samples:
                sampled_indexes.extend(o_indexes[:self.number_of_positive_samples])
            else:
                sampled_indexes.extend(o_indexes)

        #########################################
        ######### NEGATIVE SAMPLING #############
        #########################################
        # Sample negative pairs
        # Find how many negative pairs you should sample
        number_of_negative_pairs = self.number_of_samples - len(sampled_indexes)

        ########
        # DRUG #
        ########
        if ne_tags[not_this_index] == 'B-DRUG' or ne_tags[not_this_index] == 'I-DRUG':
            # '_-AE' tags
            if len(ae_indexes) >= (number_of_negative_pairs//2):
                sampled_indexes.extend(ae_indexes[:(number_of_negative_pairs//2)])
            else:
                sampled_indexes.extend(ae_indexes)
            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)

        ########
        ## AE ##
        ########
        if ne_tags[not_this_index] == 'B-AE' or ne_tags[not_this_index] == 'I-AE':
            # '_-DRUG' tags
            if len(drug_indexes) >= (number_of_negative_pairs // 2):
                sampled_indexes.extend(drug_indexes[:(number_of_negative_pairs // 2)])
            else:
                sampled_indexes.extend(drug_indexes)

            # 'O' tags
            if len(o_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(o_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(o_indexes)

        ###########
        #### O ####
        ###########
        if ne_tags[not_this_index] == 'O':
            # '_-DRUG' tags
            if len(drug_indexes) >= (number_of_negative_pairs // 2):
                sampled_indexes.extend(drug_indexes[:(number_of_negative_pairs // 2)])
            else:
                sampled_indexes.extend(drug_indexes)
            # '_-AE' tags
            if len(ae_indexes) >= (self.number_of_samples - len(sampled_indexes)):
                sampled_indexes.extend(ae_indexes[:(self.number_of_samples - len(sampled_indexes))])
            else:
                sampled_indexes.extend(ae_indexes)

        shuffle(sampled_indexes)
        return sampled_indexes


    def find_k_closest_representations_euclidean(self, representations, ne_tags, not_this_index, k):
        distances = []
        ref_represenation = representations[not_this_index]
        ref_tag = ne_tags[not_this_index]
        black_list_indexes = []
        for i, rep in enumerate(representations):
            # If the sample has the same tag then add a very large
            # distance value because you don't want to sample it.
            if ref_tag == ne_tags[i]:
                black_list_indexes.append(i)
                distances.append(1000000000)
            else:
                # Find the euclidean distance.
                distances.append(torch.dist(ref_represenation, rep, 2).item())

        # Find the indexes that would sort the distances list.
        sorted_indexes = list(np.argsort(distances))

        # Take the k-first indexes
        sampled_indexes = sorted_indexes[:k]

        # Be sure that you don't include 'positive' samples.
        sampled_indexes_final = []
        for index in sampled_indexes:
            if index not in black_list_indexes:
                sampled_indexes_final.append(index)

        return sampled_indexes_final

    def find_k_closest_representations_cosine(self, representations, ne_tags, not_this_index, k):
        # Cosine similarity.
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        similarities = []
        ref_represenation = representations[not_this_index]
        ref_tag = ne_tags[not_this_index]
        black_list_indexes = []
        for i, rep in enumerate(representations):
            # If the sample has the same tag then add a very small
            # similarity value because you don't want to sample it.
            if ref_tag == ne_tags[i]:
                black_list_indexes.append(i)
                similarities.append(-1000000000)
            else:
                # Find the cosine similarity.
                similarities.append(cos(ref_represenation, rep).item())

        # Find the indexes that would sort the similarities list in a ascending order.
        sorted_indexes = list(np.argsort(similarities))
        # Descending order
        sorted_indexes.reverse()

        # Take the k-first indexes
        sampled_indexes = sorted_indexes[:k]

        # Be sure that you don't include 'positive' samples.
        sampled_indexes_final = []
        for index in sampled_indexes:
            if index not in black_list_indexes:
                sampled_indexes_final.append(index)

        return sampled_indexes_final