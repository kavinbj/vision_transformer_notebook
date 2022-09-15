'''
Author: kavinbj
Date: 2022-09-06 22:25:15
LastEditTime: 2022-09-06 23:26:37
FilePath: resnet.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append(".")
from cnn.lenet import train_ch6
from cnn.utils import load_data_fashion_mnist

# 残差块
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def test_residual():
    blk = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)

    blk2 = Residual(3,6, use_1x1conv=True, strides=2)
    print(blk2(X).shape)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))


resnet = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

def test_resnet():
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in resnet:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

def test_resnet_train():
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch6(resnet, train_iter, test_iter, num_epochs, lr, 'cpu')
    # loss 0.117, train acc 0.954, test acc 0.866

if __name__ == '__main__':
    print('resnet')
    # test_residual()
    test_resnet()
    test_resnet_train()


