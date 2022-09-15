'''
Author: kavinbj
Date: 2022-09-10 10:55:06
LastEditTime: 2022-09-11 21:36:57
FilePath: decoder_block.py
Description: 

transformer解码器也是由多个相同的层组成。
在DecoderBlock类中实现的每个层包含了三个子层：
1、解码器自注意力、
2、“编码器-解码器”注意力
3、基于位置的前馈网络。
这些子层也都被残差连接和紧随的层规范化围绕。

在掩蔽多头解码器自注意力层（第一个子层）中，查询、键和值都来自上一个解码器层的输出。
在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；
然而，在预测阶段，其输出序列的词元是逐个生成的。
因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。
为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数dec_valid_lens，
以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算。

假设: num_layers = 4, 即有4层
forward: X state

DecoderBlock 0: state == enc_outputs, enc_valid_lens, ['None', 'None', 'None', 'None']
DecoderBlock 1: state == enc_outputs, enc_valid_lens, [torch.Size([64, 10, 32]), 'None', 'None', 'None']
DecoderBlock 2: state == enc_outputs, enc_valid_lens, [torch.Size([64, 10, 32]), torch.Size([64, 10, 32]), 'None', 'None']
DecoderBlock 3: state == enc_outputs, enc_valid_lens, [torch.Size([64, 10, 32]), torch.Size([64, 10, 32]), torch.Size([64, 10, 32]), 'None']



Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

import sys
sys.path.append(".")
from transformer.model.multihead_attention import MultiHeadAttention
from transformer.model.add_norm import AddNorm
from transformer.model.position_wise_ffn import PositionWiseFFN
from transformer.model.encoder_block import EncoderBlock

class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # print('state', self.i, [s.shape if s is not None else 'None' for s in state[2]])
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
            print('key_values', key_values.shape, 'X.shape', X.shape)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

def test_DecoderBlock():
    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24))
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    valid_lens = torch.tensor([3, 2])
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print(decoder_blk(X, state)[0].shape) 

if __name__ == '__main__':
    print('test_DecoderBlock')
    test_DecoderBlock()
