import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.netvlad import NetVLADLoupe


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):  # 论文中d_ff=1024
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class featureExtracter_RI_BEV(nn.Module):
    def __init__(self, channels=5, use_transformer=True):
        super(featureExtracter_RI_BEV, self).__init__()
        self.use_transformer = use_transformer

        # 修改卷积层以保持256维度的输出
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.conv1_add = nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(1, 1), bias=False)

        self.relu = nn.ReLU(inplace=True)

        # 修改Transformer编码器使用256维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,  # 改为256以匹配论文
            nhead=4,  # 论文中的nhead=4
            dim_feedforward=1024,  # 论文中的d_ff=1024
            activation='relu',
            batch_first=False,
            dropout=0.
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.convLast1 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.convLast2 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv1_add(out))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))

        out_1 = out.permute(0, 1, 3, 2)
        out_1 = self.relu(self.convLast1(out_1))

        if self.use_transformer:
            out = out_1.squeeze(3)
            out = out.permute(2, 0, 1)
            out = self.transformer_encoder(out)
            out = out.permute(1, 2, 0)
            out = out.unsqueeze(3)

        out = torch.cat((out_1, out), dim=1)
        out = self.relu(self.convLast2(out))
        return out


class CVTNet(nn.Module):
    def __init__(self, channels=5, use_transformer=True):
        super(CVTNet, self).__init__()
        self.use_transformer = use_transformer

        # 保持d_model=256来匹配论文
        d_model = 256
        heads = 4  # 论文中nhead=4
        dropout = 0.

        self.featureExtracter_RI = featureExtracter_RI_BEV(channels=channels, use_transformer=use_transformer)
        self.featureExtracter_BEV = featureExtracter_RI_BEV(channels=channels, use_transformer=use_transformer)

        # NetVLAD配置匹配论文
        self.net_vlad = NetVLADLoupe(
            feature_size=256,  # 修改为256以匹配论文
            max_samples=1800,
            cluster_size=64,  # 论文中dK=64
            output_dim=256,  # 论文中doutput=256
            gating=True,
            add_batch_norm=False,
            is_training=True
        )

        # 为RI和BEV分支设置独立的NetVLAD
        self.net_vlad_ri = NetVLADLoupe(
            feature_size=256,
            max_samples=900,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False,
            is_training=True
        )

        self.net_vlad_bev = NetVLADLoupe(
            feature_size=256,
            max_samples=900,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False,
            is_training=True
        )

        # Transformer相关层设置
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_2_ext = Norm(d_model)
        self.norm_3_ext = Norm(d_model)

        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = FeedForward(d_model, d_ff=1024, dropout=dropout)  # 使用论文中的d_ff=1024
        self.ff2 = FeedForward(d_model, d_ff=1024, dropout=dropout)

        self.attn1_ext = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2_ext = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1_ext = FeedForward(d_model, d_ff=1024, dropout=dropout)
        self.ff2_ext = FeedForward(d_model, d_ff=1024, dropout=dropout)

    def forward(self, x_ri_bev):
        # 分离RI和BEV输入
        x_ri = x_ri_bev[:, 0:5, :, :]  # 5通道RI输入
        x_bev = x_ri_bev[:, 5:10, :, :]  # 5通道BEV输入

        # 特征提取
        feature_ri = self.featureExtracter_RI(x_ri)
        feature_bev = self.featureExtracter_BEV(x_bev)

        # 准备特征进行融合
        feature_ri = feature_ri.squeeze(-1)
        feature_bev = feature_bev.squeeze(-1)
        feature_ri = feature_ri.permute(0, 2, 1)
        feature_bev = feature_bev.permute(0, 2, 1)
        feature_ri = F.normalize(feature_ri, dim=-1)
        feature_bev = F.normalize(feature_bev, dim=-1)

        # 应用Norm和注意力机制
        feature_ri = self.norm_1(feature_ri)
        feature_bev = self.norm_1(feature_bev)

        # 交叉注意力融合
        feature_fuse1 = feature_bev + self.attn1(feature_bev, feature_ri, feature_ri)
        feature_fuse1 = self.norm_2(feature_fuse1)
        feature_fuse1 = feature_fuse1 + self.ff1(feature_fuse1)

        feature_fuse2 = feature_ri + self.attn2(feature_ri, feature_bev, feature_bev)
        feature_fuse2 = self.norm_3(feature_fuse2)
        feature_fuse2 = feature_fuse2 + self.ff2(feature_fuse2)

        # 扩展注意力融合
        feature_fuse1_ext = feature_fuse1 + self.attn1_ext(feature_fuse1, feature_ri, feature_ri)
        feature_fuse1_ext = self.norm_2_ext(feature_fuse1_ext)
        feature_fuse1_ext = feature_fuse1_ext + self.ff1_ext(feature_fuse1_ext)

        feature_fuse2_ext = feature_fuse2 + self.attn2_ext(feature_fuse2, feature_bev, feature_bev)
        feature_fuse2_ext = self.norm_3_ext(feature_fuse2_ext)
        feature_fuse2_ext = feature_fuse2_ext + self.ff2_ext(feature_fuse2_ext)

        # 特征组合
        feature_fuse = torch.cat((feature_fuse1_ext, feature_fuse2_ext), dim=-2)
        feature_cat_origin = torch.cat((feature_bev, feature_ri), dim=-2)
        feature_fuse = torch.cat((feature_fuse, feature_cat_origin), dim=-1)

        # 准备NetVLAD处理
        feature_fuse = feature_fuse.permute(0, 2, 1)
        feature_com = feature_fuse.unsqueeze(3)

        # 应用NetVLAD
        feature_com = F.normalize(feature_com, dim=1)
        feature_com = self.net_vlad(feature_com)
        feature_com = F.normalize(feature_com, dim=1)

        # 处理RI特征
        feature_ri = feature_ri.permute(0, 2, 1)
        feature_ri = feature_ri.unsqueeze(-1)
        feature_ri_enhanced = self.net_vlad_ri(feature_ri)
        feature_ri_enhanced = F.normalize(feature_ri_enhanced, dim=1)

        # 处理BEV特征
        feature_bev = feature_bev.permute(0, 2, 1)
        feature_bev = feature_bev.unsqueeze(-1)
        feature_bev_enhanced = self.net_vlad_bev(feature_bev)
        feature_bev_enhanced = F.normalize(feature_bev_enhanced, dim=1)

        # 最终特征组合
        feature_com = torch.cat((feature_ri_enhanced, feature_com), dim=1)
        feature_com = torch.cat((feature_com, feature_bev_enhanced), dim=1)

        return feature_com