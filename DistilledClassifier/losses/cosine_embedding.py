import torch
import torch.nn as nn
import torch.nn.functional as F

class COS(nn.Module):
    '''
    Cosine Embedding loss
    '''
    def __init__(self):
        super(COS, self).__init__()

    def forward(self, fm_s, fm_t, dum):
        loss = F.cosine_embedding_loss(fm_s.view(fm_s.size(0), -1), 
                                        fm_t.view(fm_t.size(0), -1), dum)
        return loss