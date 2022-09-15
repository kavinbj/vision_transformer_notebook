<!--
 * @Author: kavinbj
 * @Date: 2022-09-14 21:26:54
 * @LastEditTime: 2022-09-15 12:31:53
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->
# Swin Transformer 学习笔记

### 《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》

论文：https://arxiv.org/abs/2103.14030v2

代码：https://github.com/microsoft/Swin-Transformer

![Swin Transformer structure](https://pic3.zhimg.com/80/v2-9a475a9b8389c48ea61da8f0b821fe56_1440w.jpg)


### transformer在cv中的挑战
1. 多尺度问题
2. 图像分辨率过大，导致序列长度过长问题

多尺度问题解决方法：
- 层级式Transformer

序列长度过长解决方法：
- 使用特征图
- 将图像打成patch
- 将图像划分成一个一个的小窗口，在窗口里面做自注意力

提出 一种包含`滑窗操作`，具有`层级设计`的Swin Transformer。

### Swin-T vs ViT
Swin Transformer 是在 Vision Transformer 的基础上使用滑动窗口（shifted windows, SW）进行改造而来。它将 Vision Transformer 中固定大小的采样快按照层次分成不同大小的块（Windows），每一个块之间的信息并不共通、独立运算从而大大提高了计算效率。从 Swin Transformer 的架构图中可以看出其与 Vision Transformer 的结构很相似，不同的点在于其采用的 Transformer Block 是由两个连续的 Swin Transformer Block 构成的，这两个 Block 块与 Vision Transformer中的 Block 块大致相同，只是将 Multi-head Self-Attention（MSA）替换成了含有不同大小 Windows 的 W-MSA 与 SW-MAS （具有滑动窗口, SW），通过 Windows 和 Shifted Windows 的 Multi-head Self-Attention 提高运算效率并最终提高分类的准确率。

![Swin-T vs ViT](https://picx.zhimg.com/80/v2-f55babd13885e3c867084ed28d0090e3_1440w.jpg?source=1940ef5c)

从 Swin Transformer 网络的整体框架图我们可以看到，首先将输入图像 I 输入到 Patch Partition 进行一个分块操作，然后送入 Linear Embedding 模块中进行通道数 channel 的调整。最后通过 stage 1, 2, 3 和 4 的特征提取和下采样得到最终的预测结果，值得注意的是每经过一个 stage，size 就会 缩小为原来的 1/2，channel 就会扩大为原来的 2 倍，与 resnet 网络类似。每个 stage 中的 Swin Transformer Block 都由两个相连的分别以 W-MSA 和 SW-MSA 为基础的 Transformer Block 构成，通过 Window 和 Shifted Window 机制提高计算性能。

### Patch Merging 模块
Patch Merging 模块将 尺寸为 H×WH×WH×W 的 Patch 块首先进行拼接并在 channel 维度上进行 concatenate 构成了 H/2×W/2×4CH/2\times W/2\times 4CH/2\times W/2\times 4C 的特征图，然后再进行 Layer Normalization 操作进行正则化，然后通过一个 Linear 层形成了一个 H/2×W/2×2CH/2 \times W/2 \times 2CH/2 \times W/2 \times 2C ，完成了特征图的下采样过程。其中 size 缩小为原来的 1/2，channel 扩大为原来的 2 倍。

![Patch Merging](https://pica.zhimg.com/80/v2-818b0a671184f4e31d568fb065b5c507_1440w.jpg?source=1940ef5c)


### W-MSA 模块
ViT 网络中的 MSA 通过 Self-Attention 使得每一个像素点都可以和其他的像素点进行内积从而得到所有像素点的信息，从而获得丰富的全局信息。但是每个像素点都需要和其他像素点进行信息交换，计算量巨大，网络的执行效率低下。因此 Swin-T 将 MSA 分个多个固定的 Windows 构成了 W-MSA，每个 Windows 之间的像素点只能与该 Windows 中的其他像素点进行内积从而获得信息，这样便大幅的减小了计算量，提高了网络的运算效率。MSA 和 W-MAS 的计算量如下所示：
Ω(MSA)=4hwC^2+2(hw)^2C
Ω(W−MSA)=4hwC^2+2M^2hwC

其中h、w和C分别代表特征图的高度、宽度和深度，M代表每个 Windows 的大小。假定 h=w=112,M=7,C=128h=w=112, M=7, C=128h=w=112, M=7, C=128 可以计算出 W-MSA 节省了

### SW-MSA 模块
虽然 W-MSA 通过划分 Windows 的方法减少了计算量，但是由于各个 Windows 之间无法进行信息的交互，因此可以看作其“感受野”缩小，无法得到较全局准确的信息从而影响网络的准确度。为了实现不同窗口之间的信息交互，我们可以将窗口滑动，偏移窗口使其包含不同的像素点，然后再进行 W-MSA 计算，将两次 W-MSA 计算的结果进行连接便可结合两个不同的 Windows 中的像素点所包含的信息从而实现 Windows 之间的信息共通。偏移窗口的 W-MSA 构成了 SW-MSA 模块，其 Windows 在 W-MSA 的基础上向右下角偏移了两个 Patch，形成了9个大小不一的块，然后使用 cyclic shift 将这 9 个块平移拼接成与 W-MSA 对应的 4 个大小相同的块，再通过 masked MSA 对这 4 个拼接块进行对应的模板计算完成信息的提取，最后通过 reverse cyclic shift 将信息数据 patch 平移回原先的位置。通过 SW-MSA 机制完成了偏移窗口的像素点的 MSA 计算并实现了不同窗口间像素点的信息交流，从而间接扩大了网络的“感受野”，提高了信息的利用效率。
![SW-MSA](https://picx.zhimg.com/80/v2-6bde53df1fc7078237a40ed1b9041574_1440w.jpg?source=1940ef5c)

### Relative position bias 机制
Swin-T 网络还在 Attention 计算中引入了相对位置偏置机制去提高网络的整体准确率表现，通过引入相对位置偏置机制，其准确度能够提高 1.2%~2.3% 不等。 以 2×2 的特征图为例，首先我们需要对特征图的各个块进行绝对位置的编号，得到每个块的绝对位置索引。然后对每个块计算其与其他块之间的相对位置，计算方法为该块的绝对位置索引减去其他块的绝对位置索引，可以得到每个块的相对位置索引矩阵。将每个块的相对位置索引矩阵展平连接构成了整个特征图的相对位置索引矩阵，具体的计算流程如下图所示。
![Relative position bias](https://pica.zhimg.com/80/v2-0577c25c8b39898968eb437eadd9a124_1440w.jpg?source=1940ef5c)

Swin-T并不是使用二维元组形式的相对位置索引矩阵，而是通过将二维元组形式的相对位置索引映射为一维的相对位置偏置（Relative position bias）来构成相应的矩阵，具体的映射方法如下：1. 将对应的相对位置行索引和列索引分别加上 M-1, 2. 将行索引和列索引分别乘以 2M-1, 3. 将行索引和列索引相加，再使用对应的相对位置偏置表（Relative position bias table）进行映射即可得到最终的相对位置偏置B。具体的计算流程如下所示：

![Relative position bias2](https://pic1.zhimg.com/80/v2-d6089751ee7d3c719da1118d7a570358_1440w.jpg?source=1940ef5c)

加入了相对位置偏置机制的 Attention 计算公式如下所示：
Attention(Q,K,V)=Softmax(QK^T/√d + B)V

其中B即为上述计算得到的相对位置偏置。

### shifted window and masked MSA
![masked MSA](https://pic1.zhimg.com/80/v2-84b7dd5ba83bf0c686a133dec758d974_1440w.jpg)

window partition --> cyclic shift --> masked MSA --> reverse cyclic shift



















