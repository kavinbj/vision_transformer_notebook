'''
Author: kavinbj
Date: 2022-09-10 00:52:07
LastEditTime: 2022-09-10 00:55:57
FilePath: masked_softmax.py
Description: 

softmax:
Softmax的含义就在于不再唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性。
通过Softmax函数就可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布。
引入指数函数的优点：
1、指数函数曲线呈现递增趋势，最重要的是斜率逐渐增大，也就是说在x轴上一个很小的变化，可以导致y轴上很大的变化。这种函数曲线能够将输出的数值拉开距离。
2、在深度学习中通常使用反向传播求解梯度进而使用梯度下降进行参数更新的过程，而指数函数在求导的时候比较方便。

引入指数函数的缺点：
当Zi值非常大的话，计算得到的数值也会变的非常大，数值可能会溢出。

masked_softmax:
softmax操作用于输出一个概率分布作为注意力权重。 在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 
例如，某些文本序列被填充了没有意义的特殊词元。 为了仅将有意义的词元作为值来获取注意力汇聚， 我们可以指定一个有效序列长度（即词元的个数）， 
以便在计算softmax时过滤掉超出指定范围的位置。 
通过这种方式，可以在masked_softmax函数中,实现这样的掩蔽softmax操作（masked softmax operation）， 其中任何超出有效长度的位置都被掩蔽并置为0。

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import math
import torch
from torch import nn

def sequence_mask(X, valid_len, value=0):
    """_summary_
    依据valid_len对二维sequence进行掩码操作
    例如：X是一个（3, 4）数据， 
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]
    valid_len 是一个shape为[3,]的list, 代表向量最大长度，超过该长度的值设为value(默认为0)
    [1，2, 3]
    结果为：
    [[1, 0, 0, 0],
     [5, 6, 0, 0],
     [9, 10, 11, 0]]
    Args:
        X (torch.tensor): shape=(seq, embed_size)
        valid_len (torch.tensor): 向量有效长度列表
        value (int, optional): Defaults to 0. 默认验码数值

    Returns:
        torch.tensor: 被掩码后的tensor
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def test_sequence_mask():
    X = torch.tensor([
        [1, 2, 3, 4], 
        [5, 6, 7, 8],
        [9, 10, 11, 12]])
    valid_len = torch.tensor([1, 2, 3])
    assert sequence_mask(X, valid_len, value=-1).equal(torch.tensor([[1, -1, -1, -1], 
                                                                     [5, 6, -1, -1], 
                                                                     [9, 10, 11, -1]]))


def masked_softmax(X, valid_lens):
    """_summary_
    通过在最后一个轴上掩蔽元素来执行softmax操作
    例如：X是一个（2，3, 4）张量， 
    [[[1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]],

      [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]]
    
    valid_lens 是一个1D张量，比如[1, 2], 其size应该与X的shape[0]一致，代表每个batch都进行相同长度的mask，比如第一个batch中，所有seq的mask长度为1，第二个batch中，所有seq的mask长度为2
    valid_lens 如果是一个2D张量，比如[[1, 2, 3], [2, 3, 4]], 其shape[0, 1] 应该与X的shape[0, 1]一致
    如：valid_lens = [[1, 2, 3], [2, 3, 4]]，
    结果为：
      [[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.2689, 0.7311, 0.0000, 0.0000],
         [0.0900, 0.2447, 0.6652, 0.0000]],

        [[0.2689, 0.7311, 0.0000, 0.0000],
         [0.0900, 0.2447, 0.6652, 0.0000],
         [0.0321, 0.0871, 0.2369, 0.6439]]]
    Args:
        X (_type_): 3D张量
        valid_lens (_type_): 1D或2D张量

    Returns:
        _type_: 验码后，在最后一个轴上进行softmax
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        # print(valid_lens.shape, valid_lens)
        # print(X.shape, X.reshape(-1, shape[-1]).shape)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def test_masked_softmax():
    """ 测试 masked_softmax """
    X = torch.tensor([[
        [1, 2, 3, 4], 
        [5, 6, 7, 8],
        [9, 10, 11, 12]],

        [[1, 2, 3, 4], 
         [5, 6, 7, 8],
         [9, 10, 11, 12]]], dtype=torch.float32)
    valid_len = torch.tensor([[1, 2, 3], [2, 3, 4]])
    res = torch.tensor([[[1.0000000, 0.0000000, 0.0000000, 0.0000000],
                       [0.2689414, 0.7310586, 0.0000000, 0.0000000],
                       [0.0900306, 0.2447285, 0.6652409, 0.0000000]],
              
                      [[0.2689414, 0.7310586, 0.0000000, 0.0000000],
                       [0.0900306, 0.2447285, 0.6652409, 0.0000000],
                       [0.0320586, 0.0871443, 0.2368828, 0.6439143]]])
  
    assert torch.equal((masked_softmax(X, valid_len)*1000).type(torch.int32), (res * 1000).type(torch.int32))



if __name__ == '__main__':
    print('test_masked_softmax')
    test_sequence_mask()
    test_masked_softmax()

