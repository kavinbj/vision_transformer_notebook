'''
Author: kavinbj
Date: 2022-09-05 23:49:11
LastEditTime: 2022-09-06 10:51:56
FilePath: cnn1.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''

import torch
from torch import nn
from d2l import torch as d2l

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)

def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = reduce_sum((X[i: i + h, j: j + w] * K))
    return Y

XX = torch.tensor([[0.0, 1.0, 2.0], 
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]])

KK = torch.tensor([[1.0, 1.0], 
                    [1.0, 1.0]])   

res = corr2d(XX, KK)

print('res', res)

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], 
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], 
                [4.0, 5.0, 6.0], 
                [7.0, 8.0, 9.0]]])

K = torch.tensor([[[0.0, 1.0], 
                   [2.0, 3.0]], 
                  [[1.0, 2.0], 
                   [3.0, 4.0]]])

print('X.shape', X.shape)
print('K.shape', K.shape)

multi_res = corr2d_multi_in(X, K)

print('multi_res', multi_res, multi_res.shape)

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)

print('K', K.shape)

multi_in_out = corr2d_multi_in_out(X, K)
print('multi_in_out', multi_in_out, multi_in_out.shape)

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))





if __name__ == '__main__':
    print('cnn1')
