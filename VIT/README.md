<!--
 * @Author: kavinbj
 * @Date: 2022-08-23 18:42:46
 * @LastEditTime: 2022-09-14 00:59:09
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->

# ViT 学习笔记

### 《AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE》
论文地址：https://arxiv.org/pdf/2010.11929.pdf

pytorch版本代码：https://github.com/lucidrains/vit-pytorch

2021年Google团队在ICLR上提出的将Transformer应用在图像分类的模型

- 当拥有足够多的数据进行预训练的时候，ViT的表现就会超过CNN，突破transformer缺少归纳偏置(局部性locality和平移等变性translation equivariance)的限制，可以在下游任务中获得较好的迁移效果
- 局部性相关性是指假设图片相邻的区域就会有相邻的特征，靠的越近的东西相关性就越强
- 平移等变性是指同样的物体不管在图片的什么地方，只要是同样的输入，遇到了同样的卷积核，就会输出同样的结果。  f(g(x)) == g(f(x)),f()代表卷积变换，g()代表线性变换。


# ViT structure
![ViT structure](https://pic1.zhimg.com/80/v2-5ac1b11cdab826232652def8e44a4828_1440w.jpg)


- Linear Projection of Flattened Patches
- Transformer Encoder
- MLP Head


### 1.1 数据预处理
H x W x C --> N x Patch_dim,  N= H/P1）x（W/P2） Patch_dim = P1 x P2 x C

e.g. 224 x 224 x 3 --> 196 x 768

### 1.2 Patch + Position Embedding
![Patch structure](https://pic4.zhimg.com/80/v2-eb0de7d9f67fd42137d5cc217b380027_1440w.jpg)


### 1.3 Transformer Encoder
![Encoder structure](https://pic1.zhimg.com/80/v2-3b4efe5632010ee57942e298eaa1dcf0_1440w.jpg)

我们的Z0首先要经过一个Layer Norm处理，在进入Multi-Head Attention层前通过变换（具体过程请查看Transformer解析）生成了Q、K、V三个向量，之后的操作与Transformer并无二致，在计算Q*K的时候我们可以把两向量内积看做计算图片块之间的关联性（与Transformer中计算词向量相似度类似），获得注意力权值后再scale到V，接着通过MLP层获得Encoder部分的输出（这里可以进行多次Encoder Block叠加，如上图所示）。

### 1.4 MLP Head
结束了Transformer Encoder，就到了我们最终的分类处理部分，在之前我们进行Encoder的时候通过concat的方式多加了一个用于分类的可学习向量，这时我们把这个向量取出来输入到MLP Head中，即经过Layer Normal --> 全连接 --> GELU --> 全连接，我们得到了最终的输出。






