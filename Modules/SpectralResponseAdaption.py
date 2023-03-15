import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from NetworkBaseModule.blocks import NetBlock


def srf_generator(SRFParam: Tensor,
                  bands: int,
                  bands_msi: int):
    """
    synthesize SRF by GMM
    """
    MU = SRFParam[:, :, 0]
    VARIANCE = SRFParam[:, :, 1]

    factors = SRFParam.shape[1]
    srf = torch.zeros([bands, bands_msi]).to(SRFParam.device)
    grid = torch.arange(bands).to(SRFParam.device)
    for i in range(bands_msi):
        for j in range(factors):
            srf[:, i] = srf[:, i].clone() + torch.exp(-(grid - MU[i, j]) ** 2 / (2 * VARIANCE[i, j] ** 2))
        srf[:, i] = srf[:, i].clone() / torch.sum(srf[:, i])

    return srf


class DenseSRF(nn.Module):
    def __init__(self,
                 DenseSRFParam: dict,
                 bands: int,
                 bands_msi: int,
                 ):
        super(DenseSRF, self).__init__()
        self.bands = bands
        self.bands_msi = bands_msi

        self.SRF = NetBlock(
            mode=DenseSRFParam["mode"],
            depth=DenseSRFParam["depth"],
            n_channels=DenseSRFParam["bands"],
            type_layers=DenseSRFParam["layers"],
            param_layers=DenseSRFParam["layers_param"]
        )

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, block_height, block_width, bands]
        :return:
        """
        [batchsize, block_height, block_width, bands] = Z.shape
        Z = Z.reshape(batchsize, block_height * block_width, bands)

        Y_r = self.SRF(Z).reshape(batchsize, block_height, block_width, self.bands_msi)

        return Y_r


class GaussianSRF(nn.Module):
    def __init__(self,
                 factors: int,
                 bands: int,
                 bands_msi: int,
                 device):
        super(GaussianSRF, self).__init__()
        self.bands = bands
        self.bands_msi = bands_msi
        self.device = device

        self.SRFParam = nn.Parameter(torch.rand(bands_msi, factors, 2))

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, block_height, block_width, bands] or [batchsize, block_height * block_width, bands]
        :return:
        """
        self.SRF = srf_generator(self.SRFParam, self.bands, self.bands_msi)

        Y_r = torch.matmul(Z, self.SRF)

        return Y_r
