import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformers_encoder.position_embedding import SinusoidalPositionalEmbedding

class MFCCEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        
        super(MFCCEmbedding, self).__init__()
        self.pos_encoder = SinusoidalPositionalEmbedding(
            embedding_dim=embedding_dim, padding_idx=0, left_pad=False
        )

    def forward(self, mfcc):

        B, T, D = mfcc.shape

        # 生成位置索引 (假设无 padding)
        positions = torch.arange(1, T + 1).unsqueeze(0).expand(B, T).to(mfcc.device).long()

        # 生成位置编码
        pos_emb = self.pos_encoder(positions)  # 形状: (B, T, D)

        # 将位置编码加到 MFCC 特征上
        mfcc_with_pos = mfcc + pos_emb  # 形状: (B, T, D)

        return mfcc_with_pos



class SpecEmbedding(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.row_embed = nn.Embedding(H, C)
        self.col_embed = nn.Embedding(W, C)

        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        i = torch.arange(H, device=x.device)
        j = torch.arange(W, device=x.device)
        x_pos = self.col_embed(j).unsqueeze(0).unsqueeze(2).expand(B, W, H, C)
        y_pos = self.row_embed(i).unsqueeze(0).unsqueeze(1).expand(B, W, H, C)
        pos = (x_pos + y_pos).permute(0, 3, 2, 1)  # [B, C, H, W]
        return x + pos
