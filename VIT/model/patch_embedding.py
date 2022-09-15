'''
Author: kavinbj
Date: 2022-09-12 20:54:59
LastEditTime: 2022-09-12 21:23:20
FilePath: patch_embedding.py
Description: 

假设原始输入的图片数据是 H x W x C，我们需要对图片进行块切割，
假设图片块大小为P1 x P2，则最终的块数量N为：N = （H/P1）x（W/P2）。
(这里需要注意H和W必须是能够被P整除的)
接下来到了图一正中间的最下面，我们看到图片块被拉成一个线性排列的序列，也就是“一维”的存在
（以此来模拟transformer中输入的词序列，即我们可以把一个图片块看做一个词），
即将切分好的图片块进行一个展平操作，那么每一个向量的长度为：Patch_dim = P1 x P2 x C。

经过上述两步操作后，我们得到了一个N x Patch_dim的输入序列。


Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
import torch
from torch import nn
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
        # 步骤一：图像分块与映射。首先将图片分块，然后接一个线性层做映射
        # from einops.layers.torch import Rearrange
        # 维度的字母可以用任何字母表示，但是中间要用空格隔开，一一对应，括号里表示相乘，维度就也相应变化
        # (batch, 3, 224, 224)->(batch, 3, ( 14 x 16), (14 x 16))->Flatten-> (batch, ( 14 x 14), ( 16 x 16 x 3 = 768))
        # 后面接上一个linear做线性映射，维度也是768

    Args:
        nn (_type_): _description_
    """
    def __init__(self, patch_height, patch_width, patch_dim, hidden_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, hidden_dim))

    def forward(self, img):
        return self.to_patch_embedding(img)


def test_PatchEmbedding():
    imgs = torch.rand(2, 3, 224, 224)

    patch_height, patch_width, channels, hidden_dim = 16, 16, 3, 768
    patch_dim = channels * patch_height * patch_width
    to_patch_embedding = PatchEmbedding(patch_height, patch_width, patch_dim, hidden_dim)
    patch_x = to_patch_embedding(imgs)
    assert patch_x.shape == torch.Size([2, 196, 768])


if __name__ == '__main__':
    print('test_PatchEmbedding')
    test_PatchEmbedding()



