'''
Author: kavinbj
Date: 2022-09-13 12:57:07
LastEditTime: 2022-09-13 14:38:55
FilePath: multihead_attention.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''

import torch
from torch import nn

import sys
sys.path.append(".")
from ViT.model.dot_product_attention import DotProductAttention


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, qkv_size, num_heads, head_dim=64, dropout=.0, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        num_hiddens = head_dim *  num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(qkv_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(qkv_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(qkv_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, X):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(X), self.num_heads)
        keys = transpose_qkv(self.W_k(X), self.num_heads)
        values = transpose_qkv(self.W_v(X), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, None)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def test_MultiHeadAttention():
    num_hiddens, num_heads = 768, 12
    attention = MultiHeadAttention(num_hiddens, num_heads, head_dim=64, dropout=.0)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
    queries = torch.ones((batch_size, num_queries, num_hiddens))
    keys = values = torch.ones((batch_size, num_kvpairs, num_hiddens))
    res = attention(queries, keys, values, valid_lens)

    assert res.shape == torch.Size([2, 4, 768])



if __name__ == '__main__':
    print('test_MultiHeadAttention')
    test_MultiHeadAttention()














