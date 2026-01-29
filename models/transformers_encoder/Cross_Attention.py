import torch
import torch.nn as nn
import torch.nn.functional as F

# without gating mechanism
# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
        
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        
#         self.to_out = nn.Linear(inner_dim, query_dim)
        
#     def forward(self, x, context):
#         h = self.heads
        
#         q = self.to_q(x)
#         k = self.to_k(context)
#         v = self.to_v(context)
        
#         q, k, v = map(lambda t: t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
        
#         # 计算注意力权重
#         sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = F.softmax(sim, dim=-1)
        
#         # 应用注意力权重
#         out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)
        
#         return self.to_out(out)

#CAFM模块。
class CAFM(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64):
        super().__init__()
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        # 输出映射
        self.to_out = nn.Linear(inner_dim, query_dim)

        # 门控层会学习融合比例
        self.gate = nn.Sequential(
            nn.Linear(query_dim + query_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
            nn.Sigmoid()
        )

    def forward(self, x, context):
        B, N, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(B, -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))

        # 注意力权重
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = F.softmax(sim, dim=-1)

        # 加权求和
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)  # [B, N, Head*dim]

        out = self.to_out(out)  # [B, N, query_dim]

        #拼接原特征与注意力特征用于 gating
        gate = self.gate(torch.cat([x, out], dim=-1))  # [B, N, query_dim]

        #门控融合。
        fused = gate * out + (1 - gate) * x

        return fused
