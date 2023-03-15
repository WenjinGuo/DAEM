from typing import Any, List

import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor


def def_loss(loss_type: str, device: Any, reduction='mean'):
    if loss_type == "l1":
        loss = nn.L1Loss(reduction=reduction)
    elif loss_type == "mse":
        loss = nn.MSELoss(reduction=reduction)
    elif loss_type == "cmd":
        loss = CMD(n_moments=5)
    elif loss_type == "sam":
        loss = SAM()
    else:
        loss = nn.L1Loss(reduction=reduction)

    loss.to(device)

    return loss


def cmd(x1: Tensor, x2: Tensor, n_moments=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.

    :param x1: shape:[batchsize, n_samples_1, n_features]
    :param x2: shape:[batchsize, n_samples_2, n_features]
    :param n_moments:
    :return:
    """
    with torch.autograd.set_detect_anomaly(True):
        mx1 = x1.mean(dim=1).unsqueeze(dim=1)
        mx2 = x2.mean(dim=1).unsqueeze(dim=1)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = l2diff(mx1, mx2)
        scms = dm
        for i in range(n_moments-1):
            scms = torch.add(scms, moment_diff(sx1, sx2, i+2))

    return scms


def l2diff(x1, x2):
    return torch.pow((x1 - x2), 2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    ss1 = torch.pow(sx1, k).mean(dim=1)
    ss2 = torch.pow(sx2, k).mean(dim=1)

    return l2diff(ss1, ss2)


class CMD(nn.Module):
    def __init__(self, n_moments=5):
        super(CMD, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1, x2):
        loss = cmd(x1, x2, self.n_moments)

        return loss


def sam(x1: Tensor, x2: Tensor):
    assert x1.shape == x2.shape
    # x1[torch.where((torch.norm(x1, 2, 1)) == 0)[0], ] += 0.0001
    # x2[torch.where((torch.norm(x2, 2, 1)) == 0)[0], ] += 0.0001
    eps = 1e-6

    sam = (x1 * x2).sum(dim=1) / ((torch.norm(x1, 2, 1) + eps) * (torch.norm(x2, 2, 1) + eps))
    sam = torch.acos(sam) * 180 / 3.14
    msam = sam.mean()

    return msam


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

    def forward(self, x1, x2):
        loss = sam(x1, x2)

        return loss