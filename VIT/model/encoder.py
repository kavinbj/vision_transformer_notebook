'''
Author: kavinbj
Date: 2022-09-13 14:40:23
LastEditTime: 2022-09-13 15:45:51
FilePath: encoder.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

import sys
sys.path.append(".")
from ViT.model.encoder_block import EncoderBlock

class ViTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, dim, num_layers, num_heads, head_dim, mlp_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                dim, num_heads, head_dim, mlp_dim, dropout=dropout, use_bias=True))
  
    def forward(self, X):
        for blk in self.blks:
            X = blk(X)
        return X


def test_ViTEncoder():
    dim, num_layers, num_heads, head_dim, mlp_dim, = 768, 12, 12, 64, 768*4
    encoder = ViTEncoder(dim, num_layers, num_heads, head_dim, mlp_dim)
    
    X = torch.rand((2, 197, 768))
    encoded_X = encoder(X)
    assert encoded_X.shape == torch.Size([2, 197, 768])


if __name__ == '__main__':
    print('test_ViTEncoder')
    test_ViTEncoder()


