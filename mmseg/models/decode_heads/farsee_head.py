import torch
import torch.nn as nn

from ..builder import HEADS
from .modules import conv1x1, DWConvBNAct, ConvBNAct
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FARSEE_HEAD(BaseDecodeHead):
    def __init__(self, high_channels, low_channels, num_class, act_type,
                 dilations=[6, 12, 18], hid_channels=256, **kwargs):
        super(FARSEE_HEAD, self).__init__(input_transform='multiple_select', **kwargs)
        # High level convolutions
        self.conv_high = nn.ModuleList([
            ConvBNAct(high_channels, hid_channels, 1, act_type=act_type)
        ])
        for dt in dilations:
            self.conv_high.append(
                nn.Sequential(
                    ConvBNAct(high_channels, hid_channels, 1, act_type=act_type),
                    DWConvBNAct(hid_channels, hid_channels, 3, dilation=dt, act_type=act_type)
                )
            )

        self.sub_pixel_high = nn.Sequential(
            conv1x1(hid_channels * 4, hid_channels * 2 * (2 ** 2)),
            nn.PixelShuffle(2)
        )

        # Low level convolutions
        self.conv_low_init = ConvBNAct(low_channels, 48, 1, act_type=act_type)
        self.conv_low = nn.ModuleList([
            ConvBNAct(hid_channels * 2 + 48, hid_channels // 2, 1, act_type=act_type)
        ])
        for dt in dilations[:-1]:
            self.conv_low.append(
                nn.Sequential(
                    ConvBNAct(hid_channels * 2 + 48, hid_channels // 2, 1, act_type=act_type),
                    DWConvBNAct(hid_channels // 2, hid_channels // 2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_low_last = nn.Sequential(
            ConvBNAct(hid_channels // 2 * 3, hid_channels * 2, 1, act_type=act_type),
            ConvBNAct(hid_channels * 2, hid_channels * 2, act_type=act_type)
        )

        self.sub_pixel_low = nn.Sequential(
            conv1x1(hid_channels * 2, num_class * (4 ** 2)),
            nn.PixelShuffle(4)
        )


    def forward(self, inputs):
        # High level features
        xmid2, xmid1, x_low, x_high = inputs
        # High level features
        high_feats = []
        for conv_high in self.conv_high:
            high_feats.append(conv_high(x_high))

        x = torch.cat(high_feats, dim=1)
        x = self.sub_pixel_high(x)

        # Low level features
        x_low = self.conv_low_init(x_low)
        x = torch.cat([x, x_low], dim=1)

        low_feats = []
        for conv_low in self.conv_low:
            low_feats.append(conv_low(x))

        x = torch.cat(low_feats, dim=1)
        x = self.conv_low_last(x)
        x = self.sub_pixel_low(x)

        return x