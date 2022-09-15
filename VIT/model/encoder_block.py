'''
Author: kavinbj
Date: 2022-09-13 13:52:28
LastEditTime: 2022-09-13 15:45:12
FilePath: encoder_block.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

import sys
sys.path.append(".")
from ViT.model.multihead_attention import MultiHeadAttention
from ViT.model.pre_norm import PreNorm
from ViT.model.feed_forward import FeedForward

class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, dim, num_heads, head_dim, mlp_dim, dropout=0., use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            dim, num_heads, head_dim = head_dim, dropout=dropout, bias=use_bias)
        self.pre_norm1 = PreNorm(dim, self.attention)
        self.ffn = FeedForward(dim, mlp_dim)
        self.pre_norm2 = PreNorm(dim, self.ffn)

    def forward(self, X):
        Y = self.pre_norm1(X)
        return self.pre_norm2(Y)

def test_EncoderBlock():
    X = torch.rand((2, 100, 768))
    encoder_blk = EncoderBlock(768, 12, 64, 768*4)
    encoder_blk.eval()
    out = encoder_blk(X)
    assert out.shape == torch.Size([2, 100, 768])


if __name__ == '__main__':
    print('test_EncoderBlock')
    test_EncoderBlock()













