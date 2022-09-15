<!--
 * @Author: kavinbj
 * @Date: 2022-09-14 21:26:54
 * @LastEditTime: 2022-09-15 12:08:47
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
![W-MSA](https://pica.zhimg.com/80/v2-818b0a671184f4e31d568fb065b5c507_1440w.jpg?source=1940ef5c)
 
其中h、w和C分别代表特征图的高度、宽度和深度，M代表每个 Windows 的大小。假定 h=w=112,M=7,C=128h=w=112, M=7, C=128h=w=112, M=7, C=128 可以计算出 W-MSA 节省了





从 Swin Transformer 网络的整体框架图我们可以看到，首先将输入图像 I 输入到 Patch Partition 进行一个分块操作，然后送入 Linear Embedding 模块中进行通道数 channel 的调整。最后通过 stage 1, 2, 3 和 4 的特征提取和下采样得到最终的预测结果，值得注意的是每经过一个 stage，size 就会 缩小为原来的 1/2，channel 就会扩大为原来的 2 倍，与 resnet 网络类似。每个 stage 中的 Swin Transformer Block 都由两个相连的分别以 W-MSA 和 SW-MSA 为基础的 Transformer Block 构成，通过 Window 和 Shifted Window 机制提高计算性能。


### 主要动机
希望实现一个层级式的Transformer，以获得类似CNN网络中FPN（特征金字塔网络）的效果，提升针对小目标检测和图像分割等任务的应用能力。为了实现层级式，给出了叫做Patch Merging的操作，从而像CNN一样，把Transformer Encoder分成了几个阶段。
H/4 x W/4 x 48 --> H/4 x W/4 x C --> H/8 x W/8 x 2C --> H/16 x W/16 x 4C --> H/32 x W/32 x 8C 
然后为了减少计算复杂度，争取能做视觉里密集预测的任务，又提出了基于窗口和移动窗口的自注意力方式，也即两个相连的blocks(Two Successive Swin Transformer Blocks)。

### shifted window and masked MSA
![masked MSA](https://pic1.zhimg.com/80/v2-84b7dd5ba83bf0c686a133dec758d974_1440w.jpg)

window partition --> cyclic shift --> masked MSA --> reverse cyclic shift



















