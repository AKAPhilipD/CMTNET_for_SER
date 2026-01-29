"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Mamba.Spec_Mamba import Spec_MambaBlock,Stem
from models.Mamba.BiMamba import BiMambaEncoder
from models.transformers_encoder.Embedding import MFCCEmbedding,SpecEmbedding
from models.transformers_encoder.Cross_Attention import CAFM


from transformers import WavLMModel


class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]

        #先处理Spectrogram模块。
        #spectrogram初始化。
        self.spec_stem=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),#[batch,64,256,384]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#[batch,64,128,192]
            SpecEmbedding(64,128,192)#[B, 64, 128, 192] #进行位置编码。
        )
        #spectrogram送入Mamba网络。
        self.spec_mamba_1=Spec_MambaBlock(in_channels=64, out_channels=128, d_state=64, d_conv=4, expand=2)#[batch,128,128,192]
        self.spec_mamba1_downsamping=nn.Sequential(
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=4, stride=4) # [batch,128,32,48]
        )
        self.spec_mamba1_res=nn.AdaptiveAvgPool2d((4,4))#[B,128,4,4]用来残差链接。

        self.spec_mamba_2=Spec_MambaBlock(in_channels=128, out_channels=256, d_state=64, d_conv=4, expand=2)#[batch,256,32,48]

        self.spec_mamba2_downsamping=nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.MaxPool2d(kernel_size=4, stride=4) # [batch,256,8,12]
        )

        self.spec_mamba_3=Spec_MambaBlock(in_channels=256, out_channels=512, d_state=64, d_conv=4, expand=2)#[Batch,512,8,12]
        self.spec_mamba3_downsamping=nn.Sequential(
            nn.InstanceNorm2d(512),
            nn.AdaptiveAvgPool2d((4,4))
        )


        #MFCC模块。
        self.mfcc_stem=nn.Sequential(
            nn.Linear(40,64),#[batch,300,64]
            nn.LayerNorm(64),
            nn.ReLU(),
            MFCCEmbedding(64)#[B, 300, 64] #进行位置编码。
        )

        #四层MFCC模块。
        self.mfcc_mamba1=BiMambaEncoder(d_model=64,n_state=64)
        self.mfcc_mamba1_downsamping=nn.Sequential(
            nn.Linear(64,128),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        )

        self.mfcc_mamba2=BiMambaEncoder(d_model=128,n_state=64)

        self.mfcc_mamba2_downsamping=nn.Sequential(
            nn.Linear(128,256),
            nn.LayerNorm(256),
            nn.Dropout(0.1)   
        )

        self.mfcc_mamba3=BiMambaEncoder(d_model=256,n_state=64)
        self.mfcc_mamba3_downsamping=nn.Sequential(
            nn.Linear(256,512),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )

        self.mfcc_mamba4=BiMambaEncoder(d_model=512,n_state=64)
        self.mfcc_mamba4_downsamping=nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )

        #注意力机制
        # Res模块。
        self.cross_att_res=CAFM(query_dim=128, key_dim=128, value_dim=128, heads=8, dim_head=64)
        #把两个res都加回去的linear机制。
        self.spec_res_linear=nn.Linear(128,512)
        self.mfcc_res_linear=nn.Linear(128,512)


        #第二轮大注意力机制。
        self.cross_att_1=CAFM(query_dim=512, key_dim=512, value_dim=512, heads=8, dim_head=64)

        #第三轮大注意力机制。
        self.spec_mfcc_res=nn.Linear(128,512)
        self.cross_att_total=CAFM(query_dim=512, key_dim=512, value_dim=512, heads=8, dim_head=64)

        #进行最后的展平工作。
        self.spec_mfcc_classifier_1 = nn.AdaptiveAvgPool1d(1)  # 将 [B, 512, 16] 转换为 [B, 512, 1]
        self.spec_mfcc_classifier_2 = nn.Sequential(
            nn.Linear(512, 256),      # 全连接层
            nn.ReLU(),                # 激活函数
            nn.Dropout(0.1),          # Dropout 防止过拟合
            nn.Linear(256, 149)         # 输出层，4 表示类别数
        )

        #调用wav模块。
        self.wav_model=WavLMModel.from_pretrained("/hy-tmp/speech_Mamba/WavLM-large")#[B,300,768]
        # for param in self.wav_model.parameters():
        #     param.requires_grad = False
        # frozen wav_model.parameters
        
        self.wav_linear_channel=nn.Linear(1024,768)
        # self.wav_linear=nn.Linear(300,149)

        #将spec_mfcc展平。
        self.spec_mfcc_flatten=nn.Flatten()

        #将wav进行sequntial
        self.wav_classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(768,149),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        #将最后进行sequential
        self.output_classifier=nn.Sequential(
            nn.Linear(298,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128,4),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
        



                                                           
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        #Spectrogram支路

       #Spectrogram支路
        audio_spec = self.spec_stem(audio_spec)
        audio_spec = self.spec_mamba_1(audio_spec)
        audio_spec = self.spec_mamba1_downsamping(audio_spec)#[B,128,32,48]

        audio_spec_res = self.spec_mamba1_res(audio_spec)#[B,128,4,4]用来参与残差连接。

        audio_spec = self.spec_mamba_2(audio_spec)#[B,512,32,48]
        audio_spec = self.spec_mamba2_downsamping(audio_spec)#[B,512,8,12]
        audio_spec = self.spec_mamba_3(audio_spec)
        audio_spec = self.spec_mamba3_downsamping(audio_spec)

        #Mfcc支路
        audio_mfcc = self.mfcc_stem(audio_mfcc)#mfcc初始化。


        audio_mfcc=self.mfcc_mamba1(audio_mfcc)#[B,300,64]
        audio_mfcc=self.mfcc_mamba1_downsamping(audio_mfcc)#[B,300,128]

        audio_mfcc=self.mfcc_mamba2(audio_mfcc)#[B,300,128]

        audio_mfcc_res=audio_mfcc #[B,300,128]送入残差链接。

        audio_mfcc=self.mfcc_mamba2_downsamping(audio_mfcc)#[B,300,256]
        audio_mfcc=self.mfcc_mamba3(audio_mfcc)
        audio_mfcc=self.mfcc_mamba3_downsamping(audio_mfcc)#[B,300,512]
        audio_mfcc=self.mfcc_mamba4(audio_mfcc)
        audio_mfcc=self.mfcc_mamba4_downsamping(audio_mfcc)#[B,300,512]
        
        

        

        #注意力机制融合。

        #spec_res预处理。
        B, C, H, W = audio_spec_res.shape  # B, 128, 4, 4
        audio_spec_res_flat = audio_spec_res.view(B, C, H * W).permute(0, 2, 1)  # → [B, 16, 128]
        
        #[B,16,128],[B,300,128]融合。
        spec_mfcc_res=self.cross_att_res(audio_spec_res_flat,audio_mfcc_res)#[B,16,128]
        
        #把两个res加回去。
        
        #先加spec_res
        audio_spec_res=self.spec_res_linear(audio_spec_res.permute(0,2,3,1))#[B,H,W,C]
        spec_output=audio_spec+(audio_spec_res.permute(0,3,1,2))#[B,512,4,4]

        #再加mfcc_res
        mfcc_output=audio_mfcc+self.mfcc_res_linear(audio_mfcc_res)#[B,300,512]

        #进入第二轮注意力机制。

        B, C, H, W = spec_output.shape  # B, 128, 4, 4
        spec_output_flat = spec_output.view(B, C, H * W).permute(0, 2, 1)  # → [B, 16, 512]
        #[B,16,512]与[B,300,512]融合。
        spec_mfcc_total=self.cross_att_1(spec_output_flat,mfcc_output)#->[B,16,512]


        #进入第三轮注意力机制。#[B,16,128] #[B,300,512]
        spec_mfcc_res=self.spec_mfcc_res(spec_mfcc_res)#->[B,16,512]
        spec_mfcc_output=self.cross_att_total(spec_mfcc_res,spec_mfcc_total) #->[B,16,512]


        spec_mfcc_output=spec_mfcc_output.permute(0,2,1)
        spec_mfcc_output=self.spec_mfcc_classifier_1(spec_mfcc_output).permute(0,2,1) #->[0,1,512]
        spec_mfcc_output=self.spec_mfcc_classifier_2(spec_mfcc_output)#->[0,1,149]


        #接下来是WAV部分。
        audio_wav=self.wav_model(audio_wav.cuda()).last_hidden_state #[B,149,1024]
        
        audio_wav=self.wav_linear_channel(audio_wav)#[B,149,768]



        audio_wav=torch.matmul(spec_mfcc_output,audio_wav)#[B,1,768]
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768]
        audio_wav = self.wav_classifier(audio_wav)#[B,149]


        #融合。
        spec_mfcc_wav=torch.cat([self.spec_mfcc_flatten(spec_mfcc_output),audio_wav],dim=-1)

        output=self.output_classifier(spec_mfcc_wav)



        output = {
            'M': output
        }  
        return output