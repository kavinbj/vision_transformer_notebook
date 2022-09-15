'''
Author: kavinbj
Date: 2022-09-05 21:54:32
LastEditTime: 2022-09-05 23:42:43
FilePath: mlp2.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''

import torch
from torch import nn
from torch.nn import functional as F


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net(X))

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print('--------------------------------------')

print([(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)

print('--------------------------------------')


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))

print(rgnet)
print(rgnet[0][1][0].bias.data)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.001)
        nn.init.zeros_(m.bias)

print('--------------------------------------')

print(net[0].weight.data[0], net[0].weight.data[0].mean(), net[0].bias.data[0])
net.apply(init_normal)
print(net[0].weight.data[0], net[0].weight.data.sum(), net[0].weight.data.mean(), net[0].bias.data[0])

print('--------------------------------------')


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

print('--------------------------------------')


if __name__ == '__main__':
    print('mlp2 mian')















