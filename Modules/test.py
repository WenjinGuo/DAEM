import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib.pyplot as plt


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


bands_msi = 3
factors = 16
bands = 31
SRFParam = torch.rand(bands_msi, factors, 2) * 20
srf = srf_generator(SRFParam, bands, bands_msi)

srf = srf.detach().numpy()
plt.plot(srf)
plt.show()
