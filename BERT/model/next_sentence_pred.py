'''
Author: kavinbj
Date: 2022-09-12 14:40:39
LastEditTime: 2022-09-12 14:47:53
FilePath: next_sentence_pred.py
Description: 

Next Sentence Prediction 下一句预测

尽管掩蔽语言建模能够编码双向上下文来表示单词，但它不能显式地建模文本对之间的逻辑关系。
为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类任务——下一句预测。
在为预训练生成句子对时，有一半的时间它们确实是标签为“真”的连续句子；
在另一半的时间里，第二个句子是从语料库中随机抽取的，标记为“假”。

NextSentencePred类使用单隐藏层的多层感知机来预测第二个句子是否是BERT输入序列中第一个句子的下一个句子。
由于Transformer编码器中的自注意力，特殊词元“<cls>”的BERT表示已经对输入的两个句子进行了编码。
因此，多层感知机分类器的输出层（self.output）以X作为输入，
其中X是多层感知机隐藏层的输出，而MLP隐藏层的输入是编码后的“<cls>”词元。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn

import sys
sys.path.append(".")
from BERT.model.encoder import BERTEncoder

class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


def test_NextSentencePred():
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
    
    # NextSentencePred实例的前向推断返回每个BERT输入序列的二分类预测。
    encoded_X = torch.flatten(encoded_X, start_dim=1)
    # NSP的输入形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    assert nsp_Y_hat.shape == torch.Size([2, 2])

    # 可以计算两个二元分类的交叉熵损失。
    nsp_y = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss(reduction='none')
    nsp_l = loss(nsp_Y_hat, nsp_y)
    assert nsp_l.shape == torch.Size([2])

if __name__ == '__main__':
    print('test_NextSentencePred')
    test_NextSentencePred()