<!--
 * @Author: kavinbj
 * @Date: 2022-08-23 18:37:28
 * @LastEditTime: 2022-09-13 22:29:26
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->
# Transformer学习笔记


## Transformer 原理解析
![ViT structure](https://pic4.zhimg.com/80/v2-0c259fb2d439b98de27d877dcd3d1fcb_1440w.jpg)


***
## Transformer 组成部分

1. embedding layer 嵌入层
2. positional encoding 位置编码
3. multi-head self-attention 多头自注意力
4. positionwise feed-forward network 基于位置的前馈网络
5. residual connection 残差连接
6. layer normalization 层规范化
7. encoder-decoder 编码器－解码器架构
8. encoder-decoder attention 编码器－解码器注意力
9. masked multi-head self-attention 掩码多头自注意力
10. auto-regressive 自回归

***


## Transformer 组成部分

Transformer的编码器是由多个相同的层叠加而成的，每个层都有两个子层（子层表示为）。第一个子层是多头自注意力（multi-head self-attention）汇聚；第二个子层是基于位置的前馈网络（positionwise feed-forward network）。

Transformer解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和层规范化。除了编码器中描述的两个子层之外，解码器还在这两个子层之间插入了第三个子层，称为编码器－解码器注意力（encoder-decoder attention）层。在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出。在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但是，解码器中的每个位置只能考虑该位置之前的所有位置。这种掩蔽（masked）注意力保留了自回归（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。





