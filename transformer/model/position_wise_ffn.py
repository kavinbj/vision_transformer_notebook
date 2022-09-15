'''
Author: kavinbj
Date: 2022-09-08 17:36:25
LastEditTime: 2022-09-08 21:18:59
FilePath: position_wise_ffn.py
Description: 
基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机(MLP),这就是称前馈网络是基于位置的(positionwise)的原因。
在本文的实现中，
输入X的形状为（批量大小，时间步数或序列长度，隐单元数或特征维度）将被一个两层的感知机，
转换成形状为（批量大小，时间步数，ffn_num_outputs）的输出张量。


Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        

def test_PositionWiseFFN():
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    result = ffn(torch.ones((2, 3, 4)))[0]
    assert result.shape == torch.Size([3, 8])
    # print(result, result.shape)

if __name__ == '__main__':
    print('test_PositionWiseFFN')
    test_PositionWiseFFN()



