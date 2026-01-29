
'''
Author: Shihe Dong
Description: BiMamba Encoder for MFCC branch.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


#audio_mfcc [B,300,40]
class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model
            d_state=n_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor
        )
        self.norm1 = RMSNorm(d_model)
        self.norm1_reverse=RMSNorm(d_model)


        #给支路分配权重的SEBlock
        self.norm2 = RMSNorm(d_model)
        self.norm2_reverse =RMSNorm(d_model)

        self.norm3 = RMSNorm(d_model)
        self.norm3_reverse =RMSNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),#原来是GELU
            nn.Linear(d_model * 4, d_model),
        )

        #定义时序卷积模块。
        self.timing_conv=nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)

        # #定义SEBlock
        # self.SEBlock = SEBlock_BiMamba(d_model, reduction=8)    

    def forward(self, x):
        reverse = torch.flip(x, dims=[1])

        # Mamba 处理

        #先给mamba初始化。
        x=self.norm1(x)
        reverse=self.norm1_reverse(reverse)

        #分别进入写好的Mamba块。并初始化。
        x_1=self.norm2(self.mamba(x))
        reverse_1 = self.norm2_reverse(self.mamba(reverse))

        #残差连接。
        x_2 = x + x_1
        reverse_2= reverse + reverse_1

        #进入feed_forward
        x_3=self.feed_forward(x_2)
        reverse_3=self.feed_forward(reverse_2)

        #残差链接。
        x = x_2 + x_3
        reverse = reverse_2 + reverse_3
        #Norm
        x=self.norm3(x)
        reverse=self.norm3_reverse(reverse)

        #把reverse翻转回来
        reverse = torch.flip(reverse, dims=[1])

        #调用时序卷积对齐
        reverse=reverse.permute(0, 2, 1)  # [B, D, L]
        reverse = self.timing_conv(reverse)
        reverse = reverse.permute(0, 2, 1)  # [B, L, D]

        output= x + reverse

        return output


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

