'''
Author: kavinbj
Date: 2022-09-13 11:50:32
LastEditTime: 2022-09-13 12:51:34
FilePath: feed_forward.py
Description: 
encoder_block中的 前馈网络

Layer Normal --> muti-head attention --> Layer Normal --> 全连接 --> GELU --> 全连接 

这里作者经过实验选取了GELU作为激活函数

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

class FeedForward(nn.Module):
    """
    dim: 维度,对于Base来说是768,Large为1024
    hidden_dim, 默认为768*4,先升维4倍再降维回去
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def test_FeedForward():
   feed_forward = FeedForward(768, 768*4)
   X = torch.rand(2, 197, 768)
   out = feed_forward(X)
   assert out.shape == torch.Size([2, 197, 768])

if __name__ == '__main__':
    print('test_FeedForward')
    test_FeedForward()
