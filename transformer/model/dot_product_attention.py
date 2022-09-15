'''
Author: kavinbj
Date: 2022-09-09 10:46:02
LastEditTime: 2022-09-10 00:57:28
FilePath: dot_product_attention.py
Description: 

点积注意力:
使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度。
 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为d。 
 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是1， 我们将点积除以√d，
 则缩放点积注意力（scaled dot-product attention）评分函数为：
 α(q, k) = qTk/√d 




Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

import sys
sys.path.append(".")
from transformer.model.masked_softmax import masked_softmax

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)

        self.attention_weights = masked_softmax(scores, valid_lens)
        # 2,3,10 * 2,10,4 = 2, 3, 4
        return torch.bmm(self.dropout(self.attention_weights), values)


def test_DotProductAttention():
    # keys、 values两个值矩阵是相同的
    keys = torch.ones((2, 10, 4))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    queries = torch.normal(0, 1, (2, 3, 4))

    attention = DotProductAttention(dropout=0)
    attention.eval()
    res = attention(queries, keys, values, valid_lens)
    assert res.shape == torch.Size([2, 3, 4]) == queries.shape


if __name__ == '__main__':
    print('test_DotProductAttention')
    test_DotProductAttention()