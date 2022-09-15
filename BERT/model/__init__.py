'''
Author: kavinbj
Date: 2022-09-12 14:48:34
LastEditTime: 2022-09-12 14:53:30
FilePath: __init__.py
Description: 

在预训练BERT时，最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合。
现在我们可以通过实例化三个类
1、BERTEncoder、
2、MaskLM
3、NextSentencePred
来定义BERTModel类。
前向推断返回编码后的BERT表示encoded_X、掩蔽语言模型预测mlm_Y_hat和下一句预测nsp_Y_hat。


BERT结合了这两个方面的优点：
它对上下文进行双向编码，并且需要对大量自然语言处理任务进行最小的架构更改。

BERT输入序列的嵌入是词元嵌入、片段嵌入和位置嵌入的和。

预训练包括两个任务：
1、掩蔽语言模型
2、下一句预测。
前者能够编码双向上下文来表示单词，而后者则显式地建模文本对之间的逻辑关系。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn
from BERT.model.encoder import BERTEncoder
from BERT.model.masked_LM import MaskLM
from BERT.model.next_sentence_pred import NextSentencePred

class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat



__all__ = [
    'BERTModel',
    'BERTEncoder',
    'MaskLM',
    'NextSentencePred'
]
