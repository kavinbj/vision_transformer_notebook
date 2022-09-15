'''
Author: kavinbj
Date: 2022-09-13 16:33:34
LastEditTime: 2022-09-13 16:42:58
FilePath: train.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

import sys
sys.path.append(".")
from ViT.model import ViT

def test_ViT():
    v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 768*4,
    pool= 'cls',
    dropout = 0.1,
    emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 224, 224)
    
    preds = v(img) # (1, 1000)

    print(preds.shape)


if __name__ == '__main__':
    print('test_ViT')
    test_ViT()













