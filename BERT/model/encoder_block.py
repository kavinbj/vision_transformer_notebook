'''
Author: kavinbj
Date: 2022-09-08 22:01:54
LastEditTime: 2022-09-12 11:51:23
FilePath: encoder_block.py
Description: 
transformer的Encoder是由多个相同的EncoderBlock叠加而成的
EncoderBlock类包含两个子层：
1、多头自注意力
2、基于位置的前馈网络
这两个子层都使用了残差连接和紧随的层规范化。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

import sys
sys.path.append(".")
from BERT.model.multihead_attention import MultiHeadAttention
from BERT.model.add_norm import AddNorm
from BERT.model.position_wise_ffn import PositionWiseFFN

class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

def test_EncoderBlock():
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    assert encoder_blk(X, valid_lens).shape == torch.Size([2, 100, 24])


if __name__ == '__main__':
    print('test_EncoderBlock')
    test_EncoderBlock()




