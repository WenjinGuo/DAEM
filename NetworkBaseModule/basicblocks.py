from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def sequential(*args: Any):
    if len(args) == 1:
        # no sequential is needed
        return args[0]

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)

    return nn.Sequential(*modules)


def dense_block(in_features=64,
                out_features=64,
                n_channels=4096,
                bias=True,
                mode='Dr',
                negative_slope=0.2):
    """
    input feature: Tensor(torch.float32) shape: [batchsize, num_features]
    """
    L= []
    for i, t in enumerate(mode):
        if t == 'D':
            L.append(
                nn.Linear(in_features=in_features,
                          out_features=out_features,
                          bias=bias))
        elif t == 'B':
            L.append(
                nn.BatchNorm1d(
                    num_features=n_channels
                ))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope,
                                  inplace=False))
        elif t == 's':
            L.append(nn.Sigmoid())
        elif t == 'S':
            L.append(nn.Softmax(dim=-1))
        elif t == "d":
            L.append(nn.Dropout(p=0.5))
        elif t == 't':
            L.append(nn.Tanh())

    return sequential(*L)


def conv_block(in_channels=64,
               out_channels=64,
               kernel_size=3,
               stride=1,
               dilation=1,
               padding=0,
               output_length=(0, 0),
               groups=1,
               bias=True,
               mode='CBR',
               device='cuda',
               negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          groups=groups,
                          padding=padding,
                          bias=bias))
        elif t == 'T':
            L.append(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   dilation=dilation,
                                   groups=groups,
                                   padding=padding,
                                   bias=bias))
        elif t == 'S':
            L.append(
                Conv2dSamePadding(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias,
                                  device=device)
            )
        elif t == 'B':
            L.append(
                nn.BatchNorm2d(out_channels,
                               momentum=0.9,
                               eps=1e-04,
                               affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope,
                                  inplace=False))
        elif t == "s":
            L.append(nn.Sigmoid())
        elif t == 'D':
            L.append(Conv2dDownSample(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      scale=stride, output_length=output_length, groups=groups, device=device))
        elif t == 'U':
            L.append(Conv2dUpSample(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    scale=stride, output_length=output_length, groups=groups, device=device))
        elif t == 'M':
            L.append(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'F':
            L.append(nn.Softmax(dim=-1))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


def conv1d_block(in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 output_length=0,
                 stride=1,
                 padding=0,
                 bias=True,
                 mode='CBR',
                 device='cuda',
                 negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias))
        elif t == 'T':
            L.append(
                nn.ConvTranspose1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias))
        elif t == 'B':
            L.append(
                nn.BatchNorm1d(out_channels,
                               momentum=0.9,
                               eps=1e-04,
                               affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm1d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope,
                                  inplace=False))
        elif t == 's':
            L.append(nn.Sigmoid())
        elif t == 'D':
            L.append(Conv1dDownSample(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      scale=stride, output_length=output_length, device=device))
        elif t == 'U':
            L.append(Conv1dUpSample(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    scale=stride, output_length=output_length, device=device))
        elif t == "S":
            L.append(Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       device=device))
        elif t == 'M':
            L.append(
                nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(
                nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class Conv2dSamePadding(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=(3, 3),
                 stride=1,
                 dilation=1,
                 bias=True,
                 groups=1,
                 padding_mode='zeros',
                 device='cuda'):
        super(Conv2dSamePadding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device

        self.conv2d = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                groups=groups,
                                dilation=self.dilation,
                                bias=self.bias)

    def forward(self, data_in: Tensor):
        height = data_in.size()[2]
        width = data_in.size()[3]
        padding_x_1, padding_x_2 = cal_padding(width, width, self.kernel_size[1], self.stride, self.dilation)
        padding_y_1, padding_y_2 = cal_padding(height, height, self.kernel_size[0], self.stride, self.dilation)
        data_padded = F.pad(data_in, [padding_x_1, padding_x_2, padding_y_1, padding_y_2]).to(device=self.device)

        data_out = self.conv2d(data_padded)

        return data_out


class ResConv2d(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=(3, 3),
                 stride=1,
                 dilation=1,
                 bias=True,
                 padding_mode='reflect',
                 mode='SrS'
                 ):
        super(ResConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode

        self.conv2d = conv_block(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 dilation=1,
                                 bias=self.bias,
                                 mode=mode)

    def forward(self, data_in: Tensor):
        data_out = self.conv2d(data_in)

        return data_out + data_in


class Conv2dDownSample(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=(3, 3),
                 scale=(1, 2),
                 output_length=(0, 0),
                 groups=1,
                 dilation=1,
                 bias=True,
                 padding_mode='reflect',
                 device='cuda'
                 ):
        super(Conv2dDownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.output_length = output_length
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device

        self.conv2d = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                dilation=dilation,
                                groups=groups,
                                stride=self.scale,
                                bias=self.bias)

    def forward(self, data_in: Tensor):
        height = data_in.size()[2]
        width = data_in.size()[3]
        # height_out = height // self.scale
        # width_out = width // self.scale
        if self.output_length[0] == 0:
            height_out = height // self.scale[0]
        else:
            height_out = self.output_length[0]
        if self.output_length[1] == 0:
            width_out = width // self.scale[1]
        else:
            width_out = self.output_length[1]
        padding_x_1, padding_x_2 = cal_padding(width, width_out, self.kernel_size[1], self.scale[1], self.dilation)
        padding_y_1, padding_y_2 = cal_padding(height, height_out, self.kernel_size[0], self.scale[0], self.dilation)
        data_padded = F.pad(data_in, [padding_x_1, padding_x_2, padding_y_1, padding_y_2], mode=self.padding_mode)\
            .to(device=self.device)

        data_out = self.conv2d(data_padded)

        return data_out


class Conv2dUpSample(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=(3, 3),
                 scale=(1, 2),
                 output_length=(0, 0),
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='reflect',
                 device='cuda',
                 is_change=False
                 ):
        super(Conv2dUpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_length = output_length

        if type(scale) == int:
            self.scale = (scale, scale)
        else:
            self.scale = scale

        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.is_change = is_change
        self.device = device

        self.conv2d = nn.ConvTranspose2d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel_size,
                                         stride=self.scale,
                                         groups=groups,
                                         dilation=self.dilation,
                                         bias=self.bias)

    def forward(self, data_in: Tensor):
        height = data_in.size()[2]
        width = data_in.size()[3]
        # height_out = height * self.scale
        # width_out = width * self.scale
        if self.output_length[0] == 0:
            height_out = height * self.scale[0]
        else:
            height_out = self.output_length[0]
        if self.output_length[1] == 0:
            width_out = width * self.scale[1]
        else:
            width_out = self.output_length[1]

        padding_x_1, padding_x_2 = cal_padding(width, width_out, self.kernel_size[1], self.scale[1], self.dilation,
                                               is_transpose=True, is_change=self.is_change)
        padding_y_1, padding_y_2 = cal_padding(height, height_out, self.kernel_size[0], self.scale[0], self.dilation,
                                               is_transpose=True, is_change=self.is_change)
        data_padded = F.pad(data_in, [padding_x_1, padding_x_2, padding_y_1, padding_y_2], mode=self.padding_mode)\
            .to(device=self.device)

        data_out = self.conv2d(data_padded)

        return data_out


class Conv1dDownSample(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=8,
                 output_length=0,
                 scale=1,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device='cuda'
                 ):
        super(Conv1dDownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.scale = scale
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device

        self.conv1d = nn.Conv1d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.scale,
                                bias=self.bias)

    def forward(self, data_in: Tensor):
        length = data_in.size()[2]
        length_out = self.output_length
        padding_1, padding_2 = cal_padding(length, length_out, self.kernel_size, self.scale, self.dilation)
        data_padded = F.pad(data_in, [padding_1, padding_2]).to(device=self.device)

        data_out = self.conv1d(data_padded)

        return data_out


class Conv1dUpSample(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=8,
                 output_length=0,
                 scale=1,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device='cuda',
                 is_change=True
                 ):
        super(Conv1dUpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.scale = scale
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.is_change = is_change
        self.device = device

        self.conv1d = nn.ConvTranspose1d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel_size,
                                         stride=self.scale,
                                         bias=self.bias)

    def forward(self, data_in: Tensor):
        length = data_in.size()[2]
        length_out = self.output_length
        padding_1, padding_2 = cal_padding(length, length_out, self.kernel_size, self.scale, self.dilation,
                                           is_transpose=True)
        data_padded = F.pad(data_in, [padding_1, padding_2]).to(self.device)

        data_out = self.conv1d(data_padded)

        return data_out


class Conv1dSamePadding(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device='cuda'):
        super(Conv1dSamePadding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device

        self.conv1d = nn.Conv1d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                dilation=self.dilation,
                                bias=self.bias)

    def forward(self, data_in: Tensor):
        length = data_in.size()[2]
        length_out = length
        padding_1, padding_2 = cal_padding(length, length_out, self.kernel_size, self.stride, self.dilation)
        data_padded = F.pad(data_in, [padding_1, padding_2]).to(device=self.device)

        data_out = self.conv1d(data_padded)

        return data_out


class ResConv1d(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=8,
                 stride=1,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 mode='SrS'
                 ):
        super(ResConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode

        self.conv1d = conv1d_block(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   bias=self.bias,
                                   mode=mode)

    def forward(self, data_in: Tensor):
        data_out = self.conv1d(data_in)

        return data_out


def cal_padding(length_in, length_out, kernel_size, stride, dilation, is_transpose=False, is_change=True):
    if not is_transpose:
        length_padded = (length_out - 1) * stride + dilation * (kernel_size - 1) + 1
        padding = length_padded - length_in

        padding_1 = padding // 2
        padding_2 = padding - padding_1
    else:
        if 0 != ((length_out - 1 - dilation * (kernel_size - 1)) % stride):
            if is_change:
                # print("WARNING! (length_out-kernel_size) should divide stride exactly.")
                temp = round((length_out - 1 - dilation * (kernel_size - 1)) / stride)
                temp = temp * stride
                kernel_size = length_out - temp
                if kernel_size <= 0:
                    temp = (length_out - 1 - dilation * (kernel_size - 1)) // stride
                    temp = temp * stride
                    kernel_size = length_out - temp
                # print("kernel_size is changed to " + str(kernel_size))
            else:
                raise Exception("Invalid value!")

        length_padded = (length_out - 1 - dilation * (kernel_size - 1)) // stride + 1
        padding = length_padded - length_in
        padding_1 = padding // 2
        padding_2 = padding - padding_1

    return padding_1, padding_2
