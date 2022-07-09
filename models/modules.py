import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys


class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(2, 2), inst_norm=True,
                 activation='leaky_relu'):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size), stride=stride,
                              padding=(kernel_size//2, kernel_size//2))
        self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True)

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            sys.exit('Activation function not supported')

        self.use_inst_norm = inst_norm

    def forward(self, x):

        out = self.conv(x)

        if self.use_inst_norm:
            out = self.inst_norm(out)

        out = self.activation(out)

        return out


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, inst_norm=True,
                 activation='leaky_relu', do_upsample=True, upsample_factor=(2, 2)):
        super().__init__()

        self.do_upsample = do_upsample

        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                              padding=(kernel_size // 2, kernel_size // 2))
        self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True)

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            sys.exit('Activation function not supported')

        self.use_inst_norm = inst_norm

    def forward(self, x, skip):

        if self.do_upsample:
            out = self.upsample(x)
        else:
            out = x

        out = torch.cat([out, skip], 1)
        out = self.conv(out)

        if self.use_inst_norm:
            out = self.inst_norm(out)

        if not(self.activation is None):
            out = self.activation(out)

        return out


class RefinementConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, inst_norm=True,
                 activation='elu'):
        super(RefinementConvLayer, self).__init__()

        self.use_inst_norm = inst_norm
        self.inst_norm = nn.InstanceNorm2d(out_ch, affine=True)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    stride=stride)

        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'linear':
            self.activation = None
        else:
            sys.exit('Activation function not supported')


    def forward(self, x):

        x = self.pad(x)
        x = self.conv_layer(x)

        if self.use_inst_norm:
            x = self.inst_norm(x)

        if not(self.activation is None):
            x = self.activation(x)

        return x

class RefinementDeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation='elu',
                 inst_norm=True, upsample='nearest'):
        super(RefinementDeconvLayer, self).__init__()

        self.use_inst_norm = inst_norm
        self.inst_norm = nn.InstanceNorm2d(out_ch, affine=True)

        self.upsample = upsample

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

        if activation == 'elu':
            self.activation = nn.ELU()
        else:
            sys.exit('Activation function not supported')

    def forward(self, x):

        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)

        x = self.pad(x)
        x = self.conv(x)

        if self.use_inst_norm:
            x = self.inst_norm(x)

        x = self.activation(x)

        return x



class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ResidualLayer, self).__init__()

        self.conv1 = RefinementConvLayer(in_ch, out_ch, kernel_size, stride, activation='elu')
        self.conv2 = RefinementConvLayer(out_ch, out_ch, kernel_size, stride, activation='elu')

    def forward(self, x):

        y = self.conv1(x)

        return self.conv2(y) + x



class Padder_Unpadder(nn.Module):

    def __init__(self, num_floors, freq_only=False):
        super().__init__()

        self.divisor = 2**num_floors
        self.vals_arr = [self.divisor*i for i in range(1, 100)]

        self.freq_only = freq_only

        self.row_pad = 0
        self.col_pad = 0

    def forward(self, x, mode):

        if mode == 'pad':

            x_rows, x_cols = x.shape[2], x.shape[3]

            new_rows = min([i for i in self.vals_arr if i >= x_rows])
            new_cols = min([i for i in self.vals_arr if i >= x_cols])

            if self.freq_only:
                x = torch.nn.functional.pad(x, (0, 0, 0, new_rows - x_rows))
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, new_rows - x_rows))
                x = torch.nn.functional.pad(x, (0, new_cols - x_cols))

            self.row_pad = new_rows - x_rows
            self.col_pad = new_cols - x_cols

        elif mode == 'unpad':

            if self.row_pad > 0:
                x = x[:, :, :-self.row_pad, :]
            if self.col_pad > 0 and not(self.freq_only):
                x = x[:, :, :, :-self.col_pad]

        return x



