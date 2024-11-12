import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from .modules import conv1x1, DWConvBNAct, ConvBNAct
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FASPP_HEAD(BaseDecodeHead):
    def __init__(self, high_channels, low_channels, mid_channels1, mid_channels2, num_class, act_type,
                 dilations=[6, 12, 18], hid_channels=256, **kwargs):
        super(FASPP_HEAD, self).__init__(input_transform='multiple_select', **kwargs)

        # High level convolutions
        dilation = 1
        self.conv_high = nn.ModuleList([
            ConvModule(
                high_channels,
                hid_channels,
                1 if dilation == 1 else 3,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        ])
        for dt in dilations:
            self.conv_high.append(
                nn.Sequential(
                    ConvModule(
                        high_channels,
                        hid_channels,
                        1,
                        dilation=dt,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels, hid_channels, 3, dilation=dt, act_type=act_type)
                )
            )

        self.sub_pixel_high = nn.Sequential(
            conv1x1(hid_channels * 4, hid_channels * 2 * (2 ** 2)),
            nn.PixelShuffle(2)
        )

        # Low level convolutions
        self.conv_low_init = ConvModule(
            low_channels,
            48,
            1,
            dilation=dilation,
            padding=0 if dilation == 1 else dilation)
        self.conv_low = nn.ModuleList([
            ConvModule(
                hid_channels * 2 + 48,
                hid_channels // 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)])
        for dt in dilations:
            self.conv_low.append(
                nn.Sequential(
                    ConvModule(
                        hid_channels * 2 + 48,
                        hid_channels // 2,
                        1,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels // 2, hid_channels // 2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_low_last = nn.Sequential(
            ConvModule(
                hid_channels // 2 * 4,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation),
            ConvModule(
                hid_channels * 2,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        )

        self.sub_pixel_low = nn.Sequential(
            conv1x1(hid_channels * 2, num_class * 2 * (4 ** 4)),
            nn.PixelShuffle(4)  # 2
        )
        # Mid2 level convolutions
        self.conv_mid2_init = ConvModule(
            mid_channels2,
            8,
            1,
            dilation=dilation,
            padding=0 if dilation == 1 else dilation)
        self.conv_mid2 = nn.ModuleList([
            ConvModule(
                8 + num_class * 32,  # 50 for greyscale, 56 for uavid input:1024, 67 for cityscapes input: 512
                hid_channels // 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        ])
        for dt in dilations[:-1]:
            self.conv_mid2.append(
                nn.Sequential(
                    ConvModule(
                        8 + num_class * 32,  # 50 for greyscale, 56 for uavid input:1024, 67 for cityscapes input: 512
                        hid_channels // 2,
                        1,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels // 2, hid_channels // 2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_mid2_last = nn.Sequential(
            ConvModule(
                hid_channels // 2 * 3,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation),
            ConvModule(
                hid_channels * 2, hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        )

        self.sub_pixel_mid2 = nn.Sequential(
            conv1x1(hid_channels * 2, num_class * (4 ** 2)),
            nn.PixelShuffle(4)
        )
    def forward(self, inputs):
        # High level features
        xmid2, xmid1, x_low, x_high = inputs
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

        # # Mid1 level features
        # xmid1 = self.conv_mid1_init(xmid1)
        #
        # x = torch.cat([x, xmid1], dim=1)
        #
        # mid1_feats = []
        # for conv_mid1 in self.conv_mid1:
        #     mid1_feats.append(conv_mid1(x))
        #
        # x = torch.cat(mid1_feats, dim=1)
        # x = self.conv_mid1_last(x)
        # x = self.sub_pixel_mid1(x)

        # Mid2 level features
        xmid2 = self.conv_mid2_init(xmid2)
        x = torch.cat([x, xmid2], dim=1)

        mid2_feats = []
        for conv_mid2 in self.conv_mid2:
            mid2_feats.append(conv_mid2(x))

        x = torch.cat(mid2_feats, dim=1)
        x = self.conv_mid2_last(x)
        x = self.sub_pixel_mid2(x)

        return x