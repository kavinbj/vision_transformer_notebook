'''
Author: kavinbj
Date: 2022-09-13 12:40:54
LastEditTime: 2022-09-13 12:49:01
FilePath: pre_norm.py
Description: 

与transformer中不同的是，这里先做LN再做Attention或者Forward， 然后加入残差连接

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

import sys
sys.path.append(".")
from ViT.model.feed_forward import FeedForward

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # 前面讲过LayerNorm（LN）
        self.fn = fn
    def forward(self, x, **kwargs):
        # 先LN, 再 fn， 最后残差连接
        return x + self.fn(self.norm(x), **kwargs)


def test_PreNorm():
    feed_forward = FeedForward(768, 768*4)
    pre_norm = PreNorm(768, feed_forward)
    X = torch.rand(2, 197, 768)
    out = pre_norm(X)

    assert out.shape == torch.Size([2, 197, 768])

if __name__ == '__main__':
    print('test_PreNorm')
    test_PreNorm()

