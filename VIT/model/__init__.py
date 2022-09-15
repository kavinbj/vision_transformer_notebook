'''
Author: kavinbj
Date: 2022-09-12 19:34:23
LastEditTime: 2022-09-13 16:42:23
FilePath: __init__.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''
from torch import nn
from ViT.model.cls_pos_concat import ClsPosConcat
from ViT.model.patch_embedding import PatchEmbedding
from ViT.model.encoder import ViTEncoder

def pair(t):
    # 判断一个t是否是tuple
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """_summary_

        Args:
            image_size (_type_): 图片尺寸
            patch_size (_type_): patch尺寸,image_size 必须能够被 patch_size整除。
            num_classes (_type_): 分类的数量
            dim (_type_): 维度,对于Base来说是768,Large为1024
            depth (_type_): Transformer Encoder模块的个数,Base为12
            heads (_type_): 多头的个数,Base=12
            mlp_dim (_type_): 多层感知机中隐藏层的神经元个数,Transformer中的FeedForward中第一个线性层升维后的维度,默认为768*4,先升维4倍再降维回去
            pool (str, optional): 选取CLS token作为输出,可选'mean',在patch维度做平均池化. Defaults to 'cls'.
            channels (int, optional): 图片输入的特征维度,RGB图像为3,灰度图为1. Defaults to 3.
            dim_head (int, optional): _description_. Defaults to 64.
            dropout (_type_, optional): Dropout几率,取值范围为[0, 1]. Defaults to 0..
            emb_dropout (_type_, optional): 进行Embedding操作时Dropout几率,取值范围为[0, 1]. Defaults to 0..
        """
        super().__init__()
        image_height, image_width = pair(image_size)    # 在这个项目中没有限定图片的尺寸
        patch_height, patch_width = pair(patch_size)    # 默认为16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        # num patches -> (224 / 16) = 14, 14 * 14 = 196
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # path dim -> 3 * 16 * 16 = 768，和Bert-base一致
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'# 输出选cls token还是做平均池化

        # 步骤一：图像分块与映射。首先将图片分块，然后接一个线性层做映射
        # from einops.layers.torch import Rearrange
        # 维度的字母可以用任何字母表示，但是中间要用空格隔开，一一对应，括号里表示相乘，维度就也相应变化
        # (batch, 3, 224, 224)->(batch, 3, ( 14 x 16), (14 x 16))->Flatten-> (batch, ( 14 x 14), ( 16 x 16 x 3 = 768))
        # 后面接上一个linear做线性映射，维度也是768
        self.to_patch_embedding = PatchEmbedding(patch_height, patch_width, patch_dim, dim)

        # pos_embedding：位置编码；cls_token：在序列最前面插入一个cls token作为分类输出
        self.cls_pos_concat = ClsPosConcat(num_patches, dim)

        # 步骤二：Transformer Encoder结构来提特征
        self.vit_encoder = ViTEncoder(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
       
        self.pool = pool
        self.to_latent = nn.Identity()# 占位操作

        # 线性层输出
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 1、图像分块
        X = self.to_patch_embedding(img)
        # 2、添加cls分类头向量和pos位置编码
        X = self.cls_pos_concat(X)
        # 3、送入transformer encoder 提取特征
        X = self.vit_encoder(X)
        print('X', X.shape)
        X = X.mean(dim = 1) if self.pool == 'mean' else X[:, 0]
        print('X', X.shape)
        X = self.to_latent(X)
        return self.mlp_head(X) # 线性输出

__all__ = [
    'ViT'
]



