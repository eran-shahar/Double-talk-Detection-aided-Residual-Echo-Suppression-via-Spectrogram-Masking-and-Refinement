from .modules import *
import torch.nn as nn
import math


class Masking_DTD(nn.Module):
    def __init__(self, in_channels=4, out_channels = [32, 64, 128, 256], fft_len=320,
                 hidden_size=128, rnn_layers=1):
        super(Masking_DTD, self).__init__()

        floors = len(out_channels)

        self.padder_unppader = Padder_Unpadder(floors)


        self.down1_dtd = DownConvBlock(in_channels=in_channels, out_channels=out_channels[0],
                                       kernel_size=3, stride=(2, 1))
        self.down2_dtd = DownConvBlock(in_channels=out_channels[0], out_channels=out_channels[1],
                                       kernel_size=3, stride=(2, 1))
        self.down3_dtd = DownConvBlock(in_channels=out_channels[1], out_channels=out_channels[2],
                                       kernel_size=3, stride=(2, 1))
        self.down4_dtd = DownConvBlock(in_channels=out_channels[2], out_channels=out_channels[3],
                                       kernel_size=3, stride=(2, 1))


        self.gru = nn.GRU(input_size=math.ceil((fft_len // 2 + 1)/(2**floors))*out_channels[-1],
                          hidden_size=hidden_size,
                          num_layers=rnn_layers,
                          batch_first=True)
        self.gru_fc = nn.Sequential(nn.Linear(hidden_size, math.ceil((fft_len // 2 + 1)/(2**floors))*out_channels[-1]),
                                    nn.PReLU())

        self.classifier = nn.Linear(hidden_size, 2)

        self.up4_dtd = UpConvBlock(in_channels=out_channels[-1]+out_channels[-2], out_channels=out_channels[-2],
                                   kernel_size=3, upsample_factor=(2, 1))
        self.up3_dtd = UpConvBlock(in_channels=out_channels[-2]+out_channels[-3], out_channels=out_channels[-3],
                                   kernel_size=3, upsample_factor=(2, 1))
        self.up2_dtd = UpConvBlock(in_channels=out_channels[-3]+out_channels[-4], out_channels=out_channels[-4],
                                   kernel_size=3, upsample_factor=(2, 1))
        self.up1_dtd = UpConvBlock(in_channels=out_channels[-4]+in_channels, out_channels=1,
                                   kernel_size=3, upsample_factor=(2, 1))

        self.down1 = DownConvBlock(in_channels=in_channels + 1, out_channels=out_channels[0],
                                   kernel_size=3, stride=(2, 2))
        self.down2 = DownConvBlock(in_channels=out_channels[0], out_channels=out_channels[1],
                                   kernel_size=3, stride=(2, 2))
        self.down3 = DownConvBlock(in_channels=out_channels[1], out_channels=out_channels[2],
                                   kernel_size=3, stride=(2, 2))
        self.down4 = DownConvBlock(in_channels=out_channels[2], out_channels=out_channels[3],
                                   kernel_size=3, stride=(2, 2))

        self.up4 = UpConvBlock(in_channels=out_channels[-1]+out_channels[-2], out_channels=out_channels[-2],
                               kernel_size=3, upsample_factor=(2, 2))
        self.up3 = UpConvBlock(in_channels=out_channels[-2]+out_channels[-3], out_channels=out_channels[-3],
                               kernel_size=3, upsample_factor=(2, 2))
        self.up2 = UpConvBlock(in_channels=out_channels[-3]+out_channels[-4], out_channels=out_channels[-4],
                               kernel_size=3, upsample_factor=(2, 2))
        self.up1 = UpConvBlock(in_channels=out_channels[-4] + in_channels + 1, out_channels=1,
                               kernel_size=3, upsample_factor=(2, 2), inst_norm=False, activation='linear')


    def forward(self, x):

        t_frames = x.shape[-1]

        x = self.padder_unppader(x, 'pad')

        out_down_1_dtd = self.down1_dtd(x)
        out_down_2_dtd = self.down2_dtd(out_down_1_dtd)
        out_down_3_dtd = self.down3_dtd(out_down_2_dtd)
        out_down_4_dtd = self.down4_dtd(out_down_3_dtd)

        batch, channel, freq, time = out_down_4_dtd.shape

        in_gru = out_down_4_dtd.reshape(batch, channel*freq, time).permute(0, 2, 1)
        out_gru, _ = self.gru(in_gru)

        out_dtd = self.classifier(out_gru)
        out_dtd = (out_dtd[:, :t_frames, ...]).permute(0, 2, 1)

        out_gru = self.gru_fc(out_gru)
        out_gru = out_gru.permute(0, 2, 1).reshape(batch, channel, freq, time)

        out_up_4_dtd = self.up4_dtd(out_gru, out_down_3_dtd)
        out_up_3_dtd = self.up3_dtd(out_up_4_dtd, out_down_2_dtd)
        out_up_2_dtd = self.up2_dtd(out_up_3_dtd, out_down_1_dtd)
        out_up_1_dtd = self.up1_dtd(out_up_2_dtd, x)

        out_up_dtd = self.padder_unppader(out_up_1_dtd, 'unpad')
        out_up_dtd = out_up_dtd.squeeze(1)

        out_down_1 = self.down1(torch.cat([x, out_up_1_dtd], 1))
        out_down_2 = self.down2(out_down_1)
        out_down_3 = self.down3(out_down_2)
        out_down_4 = self.down4(out_down_3)

        out_up_4 = self.up4(out_down_4, out_down_3)
        out_up_3 = self.up3(out_up_4, out_down_2)
        out_up_2 = self.up2(out_up_3, out_down_1)
        out_up_1 = self.up1(out_up_2, torch.cat([x, out_up_1_dtd], 1))

        out = self.padder_unppader(out_up_1, 'unpad')

        return out.squeeze(1), out_up_dtd, out_dtd


if __name__ == '__main__':

    import torch

    x = torch.randn(16, 4, 161, 201) # (batch, channels, freq., time)

    model = Masking_DTD()

    out, out_dtd_map, out_dtd = model(x)

    print(x.shape)
    print(out.shape)
    print(out_dtd.shape)
    print(out_dtd_map.shape)


