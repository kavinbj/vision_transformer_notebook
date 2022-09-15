'''
Author: kavinbj
Date: 2022-09-12 11:48:16
LastEditTime: 2022-09-12 13:39:34
FilePath: encoder.py
Description: 

BERT选择Transformer编码器作为其双向架构。
在Transformer编码器中常见是，位置嵌入被加入到输入序列的每个位置。
然而，与原始的Transformer编码器不同，BERT使用可学习的位置嵌入。
BERT输入序列的嵌入是词元嵌入、片段嵌入和位置嵌入的和。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

import sys
sys.path.append(".")
from BERT.model.encoder_block import EncoderBlock

# 将一个句子或两个句子作为输入，然后返回BERT输入序列的标记及其相应的片段索引。
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        print('tokens', tokens.shape, 'token_embedding', self.token_embedding(tokens).shape)
        print('segments', segments.shape, 'segment_embedding', self.segment_embedding(segments).shape)
        print('pos_embedding', self.pos_embedding.data.shape, self.pos_embedding.data[:, :X.shape[1], :].shape)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


def test_BERTEncoder():
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    assert encoded_X.shape == torch.Size([2, 8, 768])


if __name__ == '__main__':
    print('test_BERTEncoder')
    test_BERTEncoder()

