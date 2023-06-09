import functools
from typing import Any, Dict

import torch.nn as nn
from torch.nn import init


def init_weights(
        net: nn.Module,
        init_type: str = "xavier_uniform",
        init_bn_type: str = 'uniform',
        gain: float = 1.
):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(
        init_type, init_bn_type, gain))

    def init_fn(m: nn.Module,
                init_type: str = "orthogonal",
                init_bn_type: str = "orthogonal",
                gain: float = 1.):
        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=gain)
            m.weight.data.clamp_(-1, 1)
        elif classname.find('Conv') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data,
                                     a=0,
                                     mode='fan_in',
                                     nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data,
                                      a=0,
                                      mode='fan_in',
                                      nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError(
                    'Initialization method [{:s}] is not implemented'.format(
                        init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    'Initialization method [{:s}] is not implemented'.format(
                        init_bn_type))

            fn = functools.partial(
                init_fn,
                init_type=init_type,
                init_bn_type=init_bn_type,
                gain=gain
            )
            net.apply(fn)
