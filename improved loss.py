import torch
import torch.nn as nn

class rou2(nn.Module):
    def __init__(self, TVLoss_weight=1, w_weight=0.5):
        super(rou2, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.w_weight = w_weight

    def forward(self, x):
        h_tv = x[:, :, 1:, :] - x[:, :, :-1, :]
        w_tv = x[:, :, :, 1:] - x[:, :, :, :-1]
        h = torch.norm(h_tv, p=1)
        w = torch.norm(w_tv, p=1)
        xx = torch.norm(x, p=1)
        return self.TVLoss_weight * (h + self.w_weight * w) / (xx + 0.01)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == '__main__':
    # Rou = rou2(TVLoss_weight=0.001, w_weight=0.3).cuda()
    Rou = rou2(TVLoss_weight=0.001, w_weight=0.5)
