from masking_DTD import Masking_DTD
from modules import *


class Refinet(nn.Module):
    def __init__(self, mask_model_path, in_channels=6, floors=2):
        super(Refinet, self).__init__()

        self.in_channels = in_channels

        self.padder_unppader = Padder_Unpadder(floors)

        self.mask_model = Masking_DTD().float()
        self.mask_model.load_state_dict(torch.load(mask_model_path))

        for param in self.mask_model.parameters():
            param.requires_grad = False


        self.layers = nn.Sequential(

            RefinementConvLayer(in_ch=in_channels, out_ch=64, kernel_size=3, stride=2),
            RefinementConvLayer(in_ch=64, out_ch=128, kernel_size=3, stride=2),

            ResidualLayer(in_ch=128, out_ch=128, kernel_size=3, stride=1),
            ResidualLayer(in_ch=128, out_ch=128, kernel_size=3, stride=1),
            ResidualLayer(in_ch=128, out_ch=128, kernel_size=3, stride=1),
            ResidualLayer(in_ch=128, out_ch=128, kernel_size=3, stride=1),
            ResidualLayer(in_ch=128, out_ch=128, kernel_size=3, stride=1),

            RefinementDeconvLayer(in_ch=128, out_ch=64, kernel_size=3, stride=1),
            RefinementDeconvLayer(in_ch=64, out_ch=32, kernel_size=3, stride=1),

            RefinementConvLayer(in_ch=32, out_ch=1, kernel_size=3, stride=1, inst_norm=False,
                      activation='linear'))


    def forward(self, x):

        self.mask_model.eval()
        with torch.no_grad():
            out_mask, out_up_dtd, _ = self.mask_model(x)

        in_refine = torch.cat([x, out_mask.unsqueeze(1), out_up_dtd.unsqueeze(1)], dim=1)

        in_refine = self.padder_unppader(in_refine, 'pad')

        out_refine = self.layers(in_refine)

        out_refine = self.padder_unppader(out_refine, 'unpad')

        return out_refine.squeeze(1)
