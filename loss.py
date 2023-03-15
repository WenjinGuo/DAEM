from typing import Any, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from NetworkBaseModule.loss import def_loss


# calculate loss for optimizing param theta
def cal_loss_theta(X, Y,
                   X_r, Y_r,
                   X_ds_r,
                   X_srf, Y_psf,
                   phase,
                   loss_type_train: List[str], loss_weights_train: List[int],
                   loss_type_test: List[str], loss_weights_test: List[int],
                   device: Any):

    # X = X.reshape(batchsize, block_height * block_width, bands)\
    #     .reshape(batchsize * block_height * block_width, bands)
    # Y = Y.reshape(batchsize, block_height_msi * block_width_msi, bands_msi)\
    #     .reshape(batchsize * block_height_msi * block_width_msi, bands_msi)
    # Z = Z.reshape(batchsize, block_height_msi * block_width_msi, bands)\
    #     .reshape(batchsize * block_height_msi * block_width_msi, bands)

    X = X.permute(0, 3, 1, 2)
    Y = Y.permute(0, 3, 1, 2)

    if phase == 'train':
        # calculate loss in self-supervised stage
        loss = torch.zeros([3], dtype=torch.float32)
        loss[0] = def_loss(loss_type=loss_type_train[0], device=device)(X, X_r)
        loss[1] = def_loss(loss_type=loss_type_train[1], device=device)(Y, Y_r)
        loss[2] = def_loss(loss_type=loss_type_train[2], device=device)(X_srf, Y_psf)

        loss_all = torch.zeros([1], dtype=torch.float32)
        for i in range(len(loss)):
            loss[i] = loss[i] * loss_weights_train[i]
            loss_all = loss_all + loss[i]
    else:
        # calculate loss in testing stage for fine-tune
        loss = torch.zeros([3], dtype=torch.float32)
        loss[0] = def_loss(loss_type=loss_type_test[0], device=device)(X, X_r)
        loss[1] = def_loss(loss_type=loss_type_test[1], device=device)(Y, Y_r)
        loss[2] = def_loss(loss_type=loss_type_test[2], device=device)(X_srf, Y_psf)

        loss_all = torch.zeros([1], dtype=torch.float32)
        for i in range(len(loss)):
            loss[i] = loss[i] * loss_weights_test[i]
            loss_all = loss_all + loss[i]

    return loss, loss_all


# calculate loss for optimizing param phi
def cal_loss_phi(X,
                 X_ds_r,
                 phase,
                 loss_type_train: List[str], loss_weights_train: List[int],
                 loss_type_test: List[str], loss_weights_test: List[int],
                 device: Any):

    # X = X.reshape(batchsize, block_height * block_width, bands)\
    #     .reshape(batchsize * block_height * block_width, bands)
    # Y = Y.reshape(batchsize, block_height_msi * block_width_msi, bands_msi)\
    #     .reshape(batchsize * block_height_msi * block_width_msi, bands_msi)
    # Z = Z.reshape(batchsize, block_height_msi * block_width_msi, bands)\
    #     .reshape(batchsize * block_height_msi * block_width_msi, bands)

    X = X.permute(0, 3, 1, 2)

    if (phase == 'train') or (phase == 'test'):
        # calculate loss in self-supervised stage
        loss = torch.zeros([1], dtype=torch.float32)
        loss[0] = def_loss(loss_type=loss_type_train[0], device=device)(X_ds_r, X)

        loss_all = torch.zeros([1], dtype=torch.float32)
        for i in range(len(loss)):
            loss[i] = loss[i] * loss_weights_train[i]
            loss_all = loss_all + loss[i]
    else:
        # calculate loss in testing stage for fine-tune
        loss = torch.zeros([1], dtype=torch.float32)
        loss[0] = def_loss(loss_type=loss_type_test[0], device=device)(X_ds_r, X)

        loss_all = torch.zeros([1], dtype=torch.float32)
        for i in range(len(loss)):
            loss[i] = loss[i] * loss_weights_test[i]
            loss_all = loss_all + loss[i]

    return loss, loss_all

