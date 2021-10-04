import torch
import torch.nn as nn
import numpy as np


class InfoNCE_loss_vectorized(nn.Module):
    '''
        SimCLR loss: https://github.com/google-research/simclr // https://github.com/sthalles/SimCLR
    '''
    def __init__(self, temperature):
        super(InfoNCE_loss_vectorized, self).__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, graph_out, sent_out):
        total_loss = 0
        for g, s in zip(graph_out, sent_out):
            # Find the similarities between the sentence representation and the graph representations
            similarities = self.cos(g, s)
            similarities = similarities / self.temperature
            exp_tensor = torch.exp(similarities)
            loss = exp_tensor[0] / torch.sum(exp_tensor)
            loss = -torch.log(loss)
            total_loss = total_loss + loss

        # Divide the total loss with the number of relations
        total_loss_final = total_loss / len(graph_out)

        return total_loss_final