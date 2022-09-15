'''
Author: kavinbj
Date: 2022-09-12 13:21:08
LastEditTime: 2022-09-12 14:37:47
FilePath: masked_LM.py
Description: 

Masked Language Modeling 验码语言模型

语言模型使用左侧的上下文预测词元。
为了双向编码上下文以表示每个词元，BERT随机掩蔽词元并使用来自双向上下文的词元以自监督的方式预测掩蔽词元。此任务称为掩蔽语言模型。

在这个预训练任务中，将随机选择15%的词元作为预测的掩蔽词元。
要预测一个掩蔽词元而不使用标签作弊，一个简单的方法是总是用一个特殊的“<mask>”替换输入序列中的词元。
然而，人造特殊词元“<mask>”不会出现在微调中。
为了避免预训练和微调之间的这种不匹配，如果为预测而屏蔽词元（例如，在“this movie is great”中选择掩蔽和预测“great”），则在输入中将其替换为：

80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”）；
10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）。


我们实现了下面的MaskLM类来预测BERT预训练的掩蔽语言模型任务中的掩蔽标记。
预测使用单隐藏层的多层感知机（self.mlp）。
在前向推断中，它需要两个输入：
1、BERTEncoder的编码结果
2、用于预测的词元位置。
输出是这些位置的预测结果。

为了演示MaskLM的前向推断，我们创建了其实例mlm并对其进行了初始化。
来自BERTEncoder的正向推断encoded_X表示2个BERT输入序列。
我们将mlm_positions定义为在encoded_X的任一输入序列中预测的3个指示。
mlm的前向推断返回encoded_X的所有掩蔽位置mlm_positions处的预测结果mlm_Y_hat。
对于每个预测，结果的大小等于词表的大小。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn


import sys
sys.path.append(".")
from BERT.model.encoder import BERTEncoder

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        print('X.shape', X.shape, 'batch_idx', batch_idx, 'pred_positions', pred_positions.shape)
        print('masked_X_bf', masked_X.shape)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        print('masked_X_af', masked_X.shape)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

def test_MaskLM():
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)
    
    # 词元嵌入
    tokens = torch.randint(0, vocab_size, (2, 8))
    # 片段嵌入
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    # 正向推断结果 
    encoded_X = encoder(tokens, segments, None)
    assert encoded_X.shape == torch.Size([2, 8, 768])
    
    # Masked Language Modeling
    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)

    assert mlm_Y_hat.shape == torch.Size([2, 3, 10000])

    # 通过掩码下的预测词元mlm_Y的真实标签mlm_Y_hat，我们可以计算在BERT预训练中的遮蔽语言模型任务的交叉熵损失。
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    assert mlm_l.shape == torch.Size([6])


if __name__ == '__main__':
    print('test_MaskLM')
    test_MaskLM()

