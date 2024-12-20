import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from .modules import conv1x1, DWConvBNAct, ConvBNAct
from .decode_head import BaseDecodeHead

class MaxStyle(nn.Module):
    def __init__(self, p=0.4, mix_learnable=True, noise_learnable=True, eps=1e-6, device='gpu'):
        super(MaxStyle, self).__init__()
        self.p = p
        self.device = device
        self.mix_learnable = mix_learnable
        self.noise_learnable = noise_learnable
        self.eps = eps
        self.noise_applied = False  # Flag to monitor noise application

    def _initialize_parameters(self, input_tensor):
        device = input_tensor.device
        B, C, _, _ = input_tensor.shape
        self.gamma_noise = (
            nn.Parameter(torch.randn(B, C, 1, 1, device=device)) if self.noise_learnable else torch.zeros(B, C, 1, 1, device=device)
        )
        self.beta_noise = (
            nn.Parameter(torch.randn(B, C, 1, 1, device=device)) if self.noise_learnable else torch.zeros(B, C, 1, 1, device=device)
        )
        self.lmda = (
            nn.Parameter(torch.rand(B, 1, 1, 1, device=device)) if self.mix_learnable else torch.zeros(B, 1, 1, 1, device=device)
        )
        #logger.info("MaxStyle parameters initialized.")

    def forward(self, x):
        #logger.debug("MaxStyle forward pass started.")
        if not self.training:
            #logger.info("MaxStyle layer bypassed (evaluation mode).")
            self.noise_applied = False
            return x

        B, C, _, _ = x.shape
        if not hasattr(self, "gamma_noise") or self.gamma_noise.shape[0] != B:
            self._initialize_parameters(x)

        device = x.device
        self.perm = torch.randperm(B, device=device)

        # Randomly decide whether to apply noise for each sample in the batch
        apply_noise = torch.rand(B, device=device) < self.p
        self.noise_applied = apply_noise.any().item()  # Update flag
        apply_noise = apply_noise.view(-1, 1, 1, 1).float()  # Reshape for broadcasting

        # Calculate means and variances
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        # Mix statistics with a permuted batch
        mu2, sig2 = mu[self.perm], sig[self.perm]
        clipped_lmda = torch.clamp(self.lmda, 0, 1)
        sig_mix = sig * (1 - clipped_lmda) + sig2 * clipped_lmda
        mu_mix = mu * (1 - clipped_lmda) + mu2 * clipped_lmda

        # Apply noise
        x_aug = (sig_mix + self.gamma_noise) * x_normed + (mu_mix + self.beta_noise)

        # Use `apply_noise` to switch between augmented and original inputs
        output = x_aug * apply_noise + x * (1 - apply_noise)

        # Log whether noise was applied
        #logger.info(f"MaxStyle applied noise: {self.noise_applied}")

        return output

    def reconstruction_loss(self):
        if not hasattr(self, "gamma_noise") or self.gamma_noise is None:
            return 0.0  # Skip uninitialized layers
        loss = torch.mean(self.gamma_noise**2 + self.beta_noise**2)
        #logger.info(f"MaxStyle reconstruction loss: {loss.item()}")
        return loss

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

        # Add MaxStyle layer
        self.max_style1 = MaxStyle()

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

        # Add MaxStyle layer
        self.max_style2 = MaxStyle()

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

        # Apply Max Style
        x=self.max_style1(x)

        # Low level features
        x_low = self.conv_low_init(x_low)
        x = torch.cat([x, x_low], dim=1)

        low_feats = []
        for conv_low in self.conv_low:
            low_feats.append(conv_low(x))

        x = torch.cat(low_feats, dim=1)
        x = self.conv_low_last(x)
        x = self.sub_pixel_low(x)

        # Apply Max Style
        x=self.max_style2(x)

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