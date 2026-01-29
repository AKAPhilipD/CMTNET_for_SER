'''
Author: Shihe Dong
Description: Spec-Mamba Encoder for Spectrogram branch.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class Spec_MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=64, d_conv=4, expand=2):
        super().__init__()
        #mambablock
        self.mamba = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.linear = nn.Linear(in_channels, out_channels)  # 提升维度
        self.relu = nn.ReLU()

        self.conv2d=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        B, C, H, W = x.shape
        # 尝试删除CONV2d试一下。
        res_x=x
        res_x=self.conv2d(res_x)


        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, L, D)
        x = self.mamba(x)                            # (B, L, D)
        x = self.linear(x)                           # (B, L, out_channels)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        x = self.relu(x)
        x=x + res_x
        return x

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

# class linear_reshape(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_in = None
#         self.linear_len = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         seq_len = H * W

#         x = x.permute(0, 2, 3, 1).reshape(B, seq_len, C)  # -> (B, seq_len, C)

#         if self.linear_in is None or self.linear_in.in_features != C:
#             self.linear_in = nn.Linear(C, 512).to(x.device)

#         x = self.linear_in(x)  # -> (B, seq_len, 512)
#         x = x.transpose(1, 2)  # -> (B, 512, seq_len)

#         if self.linear_len is None or self.linear_len.in_features != seq_len:
#             self.linear_len = nn.Linear(seq_len, 32 * 48).to(x.device)

#         x = self.linear_len(x)  # -> (B, 512, 1536)
#         x = x.view(B, 512, 32, 48)

#         return x