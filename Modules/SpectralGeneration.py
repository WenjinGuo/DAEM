import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from NetworkBaseModule.blocks import NetBlock


class ConvLayerSpectralMean(nn.Module):
    def __init__(self,
                 ConvLayerSpectralParamMean: dict,
                 bands: int,
                 scale_factor: int,
                 num_endmember: int):
        super(ConvLayerSpectralMean, self).__init__()
        self.bands = bands
        self.scale_factor = scale_factor
        self.num_endmember = num_endmember

        self.SpectralGenMean = NetBlock(
            mode=ConvLayerSpectralParamMean["mode"],
            depth=ConvLayerSpectralParamMean["depth"],
            n_channels=ConvLayerSpectralParamMean["bands"],
            type_layers=ConvLayerSpectralParamMean["layers"],
            param_layers=ConvLayerSpectralParamMean["layers_param"]
        )

    def forward(self, A: Tensor):
        Z_mean = self.SpectralGenMean(A)

        return Z_mean


class ConvLayerSpectralVar(nn.Module):
    def __init__(self,
                 ConvLayerSpectralParamVar: dict,
                 bands: int,
                 scale_factor: int,
                 num_endmember: int):
        super(ConvLayerSpectralVar, self).__init__()
        self.bands = bands
        self.scale_factor = scale_factor
        self.num_endmember = num_endmember

        self.SpectralGenVar = NetBlock(
            mode=ConvLayerSpectralParamVar["mode"],
            depth=ConvLayerSpectralParamVar["depth"],
            n_channels=ConvLayerSpectralParamVar["bands"],
            type_layers=ConvLayerSpectralParamVar["layers"],
            param_layers=ConvLayerSpectralParamVar["layers_param"]
        )

    def forward(self, A: Tensor):
        Z_var = self.SpectralGenVar(A)

        return Z_var
