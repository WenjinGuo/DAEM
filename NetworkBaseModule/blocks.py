from typing import List, Any, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import NetworkBaseModule.basicblocks as B


class NetBlock(nn.Module):
    def __init__(self,
                 mode: str,
                 depth: int,
                 n_channels: int,
                 type_layers: List[str],
                 param_layers: List[Tuple[int, int]]):
        """
        :param mode: cnn or dnn
        :param depth: number of layers
        :param type_layers: type of each layer
        :param param_layers: parameters of each layer
        """

        super(NetBlock, self).__init__()

        assert depth == len(type_layers) and depth == len(param_layers), 'param in Net not match'

        if mode == 'dnn':
            self.layers = nn.ModuleList([
                B.dense_block(
                    in_features=param_layers[_][0],
                    out_features=param_layers[_][1],
                    n_channels=n_channels,
                    mode=type_layers[_]
                )
                for _ in range(depth)
            ])
        elif mode == 'conv2d':
            self.layers = nn.ModuleList([
                B.conv_block(
                    stride=param_layers[_][0],
                    in_channels=param_layers[_][1],
                    out_channels=param_layers[_][2],
                    kernel_size=[param_layers[_][3], param_layers[_][4]],
                    output_length=[param_layers[_][5], param_layers[_][6]],
                    groups=param_layers[_][7],
                    mode=type_layers[_]
                )
                for _ in range(depth)
            ])
        elif mode == 'conv1d':
            self.layers = nn.ModuleList([
                B.conv1d_block(
                    stride=param_layers[_][0],
                    in_channels=param_layers[_][1],
                    out_channels=param_layers[_][2],
                    kernel_size=param_layers[_][3],
                    output_length=param_layers[_][4],
                    mode=type_layers[_]
                )
                for _ in range(depth)
            ])
        else:
            NotImplementedError('Undefined type of layer: '.format(mode))

    def forward(self, x: Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = 0.05 * torch.randn_like(std)
    z1 = eps * std + mu

    return z1