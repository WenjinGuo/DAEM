import sys
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from NetworkBaseModule.blocks import NetBlock


def kernel_generator(Q: Tensor,
                     kernel_size: int,
                     scale_factor: int,
                     shift='center'):
    """
    modified version of https://github.com/zsyOAOA/BSRDM
    """
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(Q.device)
    M = Q * mask
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (scale_factor - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (scale_factor - 1)
    else:
        sys.exit('Please input corrected shift parameter: left , right or center!')

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(Q.device)  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ = ZZ.type(torch.float32)
    ZZ_t = ZZ.permute(0, 1, 3, 2)  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)  # k x k

    return kernel.unsqueeze(0).unsqueeze(0)


def kernel_generator_new(Q: Tensor,
                     kernel_size: int,
                     scale_factor: int,
                     shift='center'):
    """
    modified version of https://github.com/zsyOAOA/BSRDM
    """
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(Q.device)
    M = Q * mask
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (scale_factor - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (scale_factor - 1)
    else:
        sys.exit('Please input corrected shift parameter: left , right or center!')

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(Q.device)  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ = ZZ.type(torch.float32)
    ZZ_t = ZZ.permute(0, 1, 3, 2)  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)  # k x k

    return kernel.unsqueeze(0).unsqueeze(0)


class ConvLayerKernel(nn.Module):
    def __init__(self,
                 ConvLayerKernelParam: dict,
                 kernel_size: int,
                 bands: int,
                 scale_factor: int):
        super(ConvLayerKernel, self).__init__()
        self.kernel_size = kernel_size
        self.bands = bands
        self.scale_factor = scale_factor

        self.KernelAdaption = NetBlock(
            mode=ConvLayerKernelParam["mode"],
            depth=ConvLayerKernelParam["depth"],
            n_channels=ConvLayerKernelParam["bands"],
            type_layers=ConvLayerKernelParam["layers"],
            param_layers=ConvLayerKernelParam["layers_param"]
        )

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, block_height, block_width, bands]
        :return:
        """
        X_r = self.KernelAdaption(Z.permute(0, 3, 1, 2))

        return X_r


class EntireKernel(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 bands: int,
                 scale_factor: int):
        super(EntireKernel, self).__init__()
        self.kernel_size = kernel_size
        self.bands = bands
        self.scale_factor = scale_factor
        self.KernelAdaption = nn.Parameter(torch.randn(self.scale_factor, self.scale_factor))

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, bands, block_height, block_width]
        :return:
        """
        [batchsize, bands, block_height, block_width] = Z.shape
        PSF = F.softmax(self.KernelAdaption.reshape(self.scale_factor * self.scale_factor))\
            .reshape(self.scale_factor, self.scale_factor)
        self.psf = PSF
        X_r = F.conv2d(Z, PSF.repeat(bands, 1, 1, 1), groups=bands)
        X_r = X_r[:, :, 0::self.scale_factor, 0::self.scale_factor]

        return X_r


class GaussianKernel(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 bands: int,
                 scale_factor: int):
        super(GaussianKernel, self).__init__()
        self.kernel_size = kernel_size
        self.bands = bands
        self.scale_factor = scale_factor

        self.KernelParam = nn.Parameter(5 * torch.eye(2, 2))

    def re_init(self):
        self.KernelParam = nn.Parameter(5 * torch.eye(2, 2))

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, bands, block_height, block_width]
        :return:
        """
        [batchsize, bands, block_height, block_width] = Z.shape
        # Noise = torch.randn(self.KernelParam.shape).to(device=self.KernelParam.device)
        # self.KernelAdaption = kernel_generator((self.KernelParam + Noise * 0.1),
        #                                        self.kernel_size, self.scale_factor, shift='center')
        self.KernelAdaption = kernel_generator(self.KernelParam,
                                               self.kernel_size, self.scale_factor, shift='center')
        # kernel_gt = sio.loadmat('D:/personal/implements/algorithm/Degradation Adaption on HSI-SR/Degradation_params'
        #                         '/blur_kernel/5.mat')['data']
        # self.KernelAdaption = nn.Parameter(torch.from_numpy(kernel_gt.astype(np.float32))).to(device='cuda')
        X_r = F.conv2d(Z, self.KernelAdaption.repeat(bands, 1, 1, 1), groups=bands)
        X_r = X_r[:, :, 0::self.scale_factor, 0::self.scale_factor]

        return X_r


class GaussianKernel_new(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 bands: int,
                 scale_factor: int):
        super(GaussianKernel_new, self).__init__()
        self.kernel_size = kernel_size
        self.bands = bands
        self.scale_factor = scale_factor

        self.KernelParam = nn.Parameter(5 * torch.eye(2, 2))

    def forward(self, Z: Tensor):
        """
        :param Z: [batchsize, bands, block_height, block_width]
        :return:
        """
        [batchsize, bands, block_height, block_width] = Z.shape
        # Noise = torch.randn(self.KernelParam.shape).to(device=self.KernelParam.device)
        # self.KernelAdaption = kernel_generator((self.KernelParam + Noise * 0.1),
        #                                        self.kernel_size, self.scale_factor, shift='center')
        self.KernelAdaption = kernel_generator(self.KernelParam,
                                               self.kernel_size, self.scale_factor, shift='center')
        X_r = F.conv2d(Z, self.KernelAdaption.repeat(bands, 1, 1, 1), groups=bands)
        X_r = X_r[:, :, 0::self.scale_factor, 0::self.scale_factor]

        return X_r