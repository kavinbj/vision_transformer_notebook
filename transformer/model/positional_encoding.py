'''
Author: kavinbj
Date: 2022-09-08 21:23:23
LastEditTime: 2022-09-08 21:59:21
FilePath: positional_encoding.py
Description: 
在处理词元序列时，循环神经网络是逐个的重复地处理词元的， 而自注意力则因为并行计算而放弃了顺序操作。 
为了使用序列的顺序信息，我们通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。 
位置编码可以通过学习得到也可以直接固定得到。 这里我们描述的是基于正弦函数和余弦函数的固定位置编码

1、跟CNN/RNN不同，自注意力并没有记录位置信息。
2、位置编码将位置信息注入到输入里

绝对位置编码：
类似于计算机使用的二进制编码
0  =  000
1  =  001
2  =  010
3  =  011
4  =  100
5  =  101
6  =  110
7  =  111

相对位置编码：
1、位置于 i + δ 的位置编码可以通过线性投影位置i处的位置编码来表示
即 Pi+δ = W * Pi



Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn
from matplotlib import pyplot as plt

import sys
sys.path.append(".")
from transformer.model.utils import plot

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def test_positionalEncoding():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    # print('P', P.shape, P[0, :, 0:32].shape)
    plot(torch.arange(num_steps), P[0, :, 8:12].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(0, 4)])
    plt.show()

if __name__ == '__main__':
    print('test_positionalEncoding')
    test_positionalEncoding()

