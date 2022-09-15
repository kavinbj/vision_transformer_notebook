'''
Author: kavinbj
Date: 2022-09-12 21:49:39
LastEditTime: 2022-09-13 16:19:51
FilePath: cls_pos_concat.py
Description: 


cls: 我们采取谁作为最终分类头的输入？
这里增加了一个可学习的Xcls，以此来作为最终输入分类头的向量，
通过concat的方式与原一维图片块向量(N x Patch_dim)进行拼接（故维度为[1，1，dim])
e.g. N为196，dim为768， 因此合并后的维度为(B, 196+1, 768)

pos：在Transformer中我们知道，词序列在输入时加入了一种位置编码信息（即Positional encoding），
同样在Vision Transformer中，为了尽可能贴合原Transformer中encoder部分，
也加入了一种位置信息，不过以一个可学习的变量来代替(参考BERT)，维度为：[1, N + 1, dim]
（此处N即为上文图片块总数N，N+1为加了后的总数），通过逐元素加和（element-add）的方式添加到原一维图片块向量中去。


Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn


class ClsPosConcat(nn.Module):
    def __init__(self, num_patches, hidden_dim, emb_dropout=.0, **kwargs):
        super().__init__(**kwargs)
        # pos_embedding：位置编码；cls_token：在序列最前面插入一个cls token作为分类输出
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, X):
        batch, num, _ = X.shape
        # 1 x 1 x 768的CLS token重复至 batch x 1 x 768
        cls_tokens = self.cls_token.repeat(batch, 1, 1)
        # 拼接操作 将cls_token展平拼接
        y = torch.cat((cls_tokens, X), dim=1)
        # 位置编码，因为多了个CLS token所以要n+1
        y += self.pos_embedding[:, :(num + 1)]
        # x.shape -> (batch, 196 + 1, 768)
        y = self.dropout(y)
        return y

def test_ClsPosConcat():
    num_patches, hidden_dim = 196, 768
    cls_pos_concat = ClsPosConcat(num_patches, hidden_dim)
    
    X = torch.rand(2, 196, 768)
    out = cls_pos_concat(X)
    assert out.shape == torch.Size([2, 197, 768])

if __name__ == '__main__':
    print('test_ClsPosConcat')
    test_ClsPosConcat()

