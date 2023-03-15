from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from Modules.AbundanceEstimation import ConvLayerAbundance
from Modules.SpectralGeneration import ConvLayerSpectralMean, ConvLayerSpectralVar
from Modules.SpectralResponseAdaption import GaussianSRF
from Modules.KernelAdaption import GaussianKernel


def reparameterize(mu, std):
    eps = torch.randn_like(std)
    z1 = eps * std + mu

    return z1


class DMFNet(nn.Module):
    def __init__(self,
                 ConvLayerAbundanceParam: dict,
                 ConvLayerSpectralParamMean: dict,
                 ConvLayerSpectralParamVar: dict,
                 bands: int,
                 bands_msi: int,
                 scale_factor: int,
                 num_endmember: int,
                 factors: int,
                 device):
        super(DMFNet, self).__init__()

        self.bands = bands
        self.bands_msi = bands_msi
        self.scale_factor = scale_factor
        self.num_endmember = num_endmember

        self.AbundanceEstNet = ConvLayerAbundance(ConvLayerAbundanceParam, self.bands, self.scale_factor)
        self.SpectralMeanEstNet = ConvLayerSpectralMean(ConvLayerSpectralParamMean, self.bands,
                                                        self.scale_factor, self.num_endmember)
        self.SpectralVarEstNet = ConvLayerSpectralVar(ConvLayerSpectralParamVar, self.bands,
                                                      self.scale_factor, self.num_endmember)
        self.SRFEst = GaussianSRF(factors, self.bands, self.bands_msi, device)
        self.PSFEst = GaussianKernel(self.scale_factor, self.bands, self.scale_factor)

        self.device = device

    def forward(self, X: Tensor, Y: Tensor, A_ds=None, stage='train'):
        """
        :param stage: string
        :param A_ds: [batchsize, block_height, block_width, self.num_endmember]
        :param X: [batchsize, block_height, block_width, bands]
        :param Y: [batchsize, block_height_msi, block_width_msi, bands_msi]
        :param phase:
        :return:
        """
        block_height_msi, block_width_msi = Y.shape[1], Y.shape[2]
        batchsize, block_height, block_width, bands = X.shape

        if (stage == "train") or (stage == "test"):
            # E-step
            A = self.AbundanceEstNet(X, Y, scale_factor=self.scale_factor)
            A = A.reshape(batchsize, self.num_endmember, block_height_msi * block_width_msi).permute(0, 2, 1)

            Z_mean = self.SpectralMeanEstNet(A)
            Z_var = self.SpectralVarEstNet(A)

            Z_r = reparameterize(Z_mean, Z_var)
            Z_r = Z_r.reshape(batchsize, block_height_msi, block_width_msi, bands).permute(0, 3, 1, 2)
            # Z_r.requires_grad = False

            # calculate variables for M-step
            X_r = self.PSFEst(Z_r)
            Y_r = self.SRFEst(Z_r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            Y_ds = self.PSFEst(Y.permute(0, 3, 1, 2))
            A_ds = self.AbundanceEstNet(X, Y_ds.permute(0, 2, 3, 1), scale_factor=self.scale_factor) \
                .reshape(batchsize, self.num_endmember,
                         block_height_msi // self.scale_factor * block_width_msi // self.scale_factor).permute(0, 2, 1)
            Z_mean_ds = self.SpectralMeanEstNet(A_ds)
            Z_var_ds = self.SpectralVarEstNet(A_ds)
            X_ds_r = reparameterize(Z_mean_ds, Z_var_ds)
            X_ds_r = X_ds_r.reshape(batchsize, block_height, block_width, bands).permute(0, 3, 1, 2)

            X_psf = self.SRFEst(X_r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            Y_srf = self.PSFEst(Y_r)

            # adjust each variable to 4-dim tensor
            A = A.reshape(batchsize, block_height_msi, block_width_msi, self.num_endmember).permute(0, 3, 1, 2)
            A_ds = A_ds.reshape(batchsize, block_height, block_width, self.num_endmember).permute(0, 3, 1, 2)
        elif stage == "fine-tune":
            # # only calculate spectral networks
            A_ds = A_ds.reshape(batchsize, self.num_endmember, block_height * block_width).permute(0, 2, 1)
            Z_mean_ds = self.SpectralMeanEstNet(A_ds)
            Z_var_ds = self.SpectralVarEstNet(A_ds)

            X_ds_r = reparameterize(Z_mean_ds, Z_var_ds)
            X_ds_r = X_ds_r.reshape(batchsize, block_height, block_width, bands).permute(0, 3, 1, 2)

            A, Z_r, X_r, Y_r, X_psf, Y_srf = None, None, None, None, None, None
        else:
            raise Exception('Wrong mode, please check in')

        return A_ds, A, Z_r, X_r, Y_r, X_ds_r, X_psf, Y_srf

    def forward_opt_phi(self, X: Tensor, Y: Tensor):
        """
        :param X: [batchsize, block_height, block_width, bands]
        :param Y: [batchsize, block_height_msi, block_width_msi, bands_msi]
        :return:
        """
        block_height_msi, block_width_msi = Y.shape[1], Y.shape[2]
        batchsize, block_height, block_width, bands = X.shape

        Y_ds = self.PSFEst(Y.permute(0, 3, 1, 2))
        A_ds = self.AbundanceEstNet(X, Y_ds.permute(0, 2, 3, 1), scale_factor=self.scale_factor) \
            .reshape(batchsize, self.num_endmember,
                     block_height_msi // self.scale_factor * block_width_msi // self.scale_factor).permute(0, 2, 1)
        Z_mean_ds = self.SpectralMeanEstNet(A_ds)
        Z_var_ds = self.SpectralVarEstNet(A_ds)
        X_ds_r = reparameterize(Z_mean_ds, Z_var_ds)
        X_ds_r = X_ds_r.reshape(batchsize, block_height, block_width, bands).permute(0, 3, 1, 2)

        return X_ds_r
