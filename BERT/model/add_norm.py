'''
Author: kavinbj
Date: 2022-09-08 18:19:12
LastEditTime: 2022-09-11 15:09:57
FilePath: add_norm.py
Description: 

在一个小批量的样本内基于批量规范化对数据进行重新中心化和重新缩放的调整。
layer normalization 层规范化和 batch normalization 批量规范化的目标相同，但LN层规范化是基于特征维度进行规范化。
尽管BN批量规范化在计算机视觉中被广泛应用，但在自然语言处理任务中（输入通常是变长序列）BN批量规范化通常不如LN层规范化的效果好。

BN: 固定小批量里面的均值和方差，然后在做额外的调整（可学习的参数）
解决的问题是：
1、损失出现在最后，后面的层训练较快
2、数据在最底层，底部的层训练较慢，所有都跟着变，最后的那些层需要重新学习多次，导致收敛很慢

Xi+1 = γ(Xi - μB)/σB + β
可学习的参数为  γ, β
作用在：
1、全连接层和卷积层输出上，激活函数前，（相当于一个线性变换，所以激活函数放在BN后面进行非线性）
2、全连接层和卷积层输入上。（相当于给输入做一个线性变换，使得基于固定均值和方差进行矫正）
对于全连接层，作用在特征维。（比如768x10, 作用于10特征向量上，进行基于固定均值和方差的矫正）
对于卷积层，作用在通道维。（可以认为（batch, w, h, channel）, channel为每个像素的特征向量，对这个向量进行基于固定均值和方差的矫正）

BN在做什么：
1、减少内部协变量转移
2、后续有论文支出它可能就是通过在每个小批量里加入噪音来控制模型复杂度
因此没必要跟Drop out混合使用

总结：
1、批量归一化固定小批量中的均值和方差，然后学习出适合的偏移和缩放
2、可以加快收敛速度，但一般不改变模型精度

LN: 是在每一个样本(一个样本里的不同通道)上计算均值和方差，(Batch，Len，Dimension)。计算Len和Dimension
BN：在每一层的每一批数据(一个batch里的同一通道)上进行归一化。(Batch，Len，Dimension)，累计计算 Dimension

BN缺点
* Btz太小会影响。对batchsize的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布；
* BN实际使用时需要计算并且保存某一层神经网络batch的均值和方差等统计信息，对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其他sequence长很多，这样training时，计算很麻烦
BN不适用于RNN等动态网络，不适合序列长度会变的NLP应用，会不稳定，适用于CNN；LN适用于RNN。



Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

def test_ln_bn():
    ln = nn.LayerNorm(2)
    bn = nn.BatchNorm1d(2)
    X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    # 在训练模式下计算X的均值和方差
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

def test_AddNorm():
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    result = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
    assert result == torch.Size([2, 3, 4])

if __name__ == '__main__':
    print('test_AddNorm')
    test_ln_bn()
    test_AddNorm()

