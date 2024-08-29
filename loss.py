# https://github.com/yinguobing/arcface/blob/main/losses.py)

from math import pi
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ArcLoss(nn.Module):
    """ 
        Additive angular margin loss
    """
    def __init__(self, in_features, out_features, margin=0.5, scale=64):
        super(ArcLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

        self.margin = margin
        self.scale = scale
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.threshold = torch.cos(torch.tensor(pi) - torch.tensor(margin))
        self.safe_margin = self.sin_m * margin

    def forward(self, embeddings, labels):
        # cos(theta)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_t = embeddings @ weight_norm.T
        sin_t = torch.sqrt(1.0-torch.pow(cos_t,2))

        # cos(theta + m)
        cos_t_margin = torch.where(cos_t > self.threshold,
                                    cos_t*self.cos_m - sin_t*self.sin_m,
                                    cos_t - self.safe_margin)

        # the labels here are assumed to be one-hot encoded.
        mask = labels 
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale

        # compute the softmax cross-entropy loss
        losses = F.cross_entropy(logits, mask)

        return losses