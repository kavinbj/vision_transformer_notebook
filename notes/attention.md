# attention 学习笔记

参考文章：
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


## Attention注意力机制与self-attention自注意力机制

### 1.为什么要因为注意力机制

在Attention诞生之前，已经有CNN和RNN及其变体模型了，那为什么还要引入attention机制？主要有两个方面的原因，如下：

**（1）计算能力的限制**：当要记住很多“信息“，模型就要变得更复杂，然而目前计算能力依然是限制神经网络发展的瓶颈。

**（2）优化算法的限制**：LSTM只能在一定程度上缓解RNN中的长距离依赖问题，且信息“记忆”能力并不高。

### 2.什么是注意力机制

在介绍什么是注意力机制之前，先让大家看一张图片。当大家看到下面图片，会首先看到什么内容？当过载信息映入眼帘时，我们的大脑会把注意力放在主要的信息上，这就是大脑的注意力机制。

![attention](https://pic4.zhimg.com/80/v2-a9bf4f074f460a501ecbe4e5c68a7aff_1440w.jpg#pic_center)


同样，当我们读一句话时，大脑也会首先记住重要的词汇，这样就可以把注意力机制应用到自然语言处理任务中，于是人们就通过借助人脑处理信息过载的方式，提出了Attention机制

### 3.注意力机制模型
![attention model](https://pic2.zhimg.com/80/v2-b3da4bb2f08f7b2cd61c10353c502459_1440w.jpg#pic_center)

从本质上理解，Attention是从大量信息中有筛选出少量重要信息，并聚焦到这些重要信息上，忽略大多不重要的信息。权重越大越聚焦于其对应的Value值上，即权重代表了信息的重要性，而Value是其对应的信息。

至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：**第一个过程是根据Query和Key计算权重系数，第二个过程根据权重系数对Value进行加权求和。** 而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如图展示的三个阶段。

![attention steps](https://pic4.zhimg.com/80/v2-99c73a55cee546d47549cdfd0946adf7_1440w.jpg#pic_center)

在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个 Keyi ，计算两者的相似性或者相关性，最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值，即如下方式：

![attention dot product](https://pic2.zhimg.com/80/v2-a88f90dcd63f76951d70f90b10bd8d75_1440w.jpg#pic_center)

第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。即一般采用如下公式计算：

![attention softmax](https://pic4.zhimg.com/80/v2-89d3eaf593653191445b688bf1a22b8b_1440w.jpg#pic_center)

第二阶段的计算结果 ai 即为 Valuei 对应的权重系数，然后进行加权求和即可得到Attention数值：

![attention value](https://pic1.zhimg.com/80/v2-f8ba30c7099292cda390b7f4e9f09b9c_1440w.jpg#pic_center)

通过如上三个阶段的计算，即可求出针对Query的Attention数值，目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程。

### 4.Self-attention自注意力机制
自注意力机制是注意力机制的变体，其减少了对外部信息的依赖，更擅长捕捉数据或特征的内部相关性。

自注意力机制在文本中的应用，主要是通过计算单词间的互相影响，来解决长距离依赖问题。

自注意力机制的计算过程：

1.将输入单词转化成嵌入向量；

2.根据嵌入向量得到q，k，v三个向量；

3.为每个向量计算一个score：score =q . k ；

4.为了梯度的稳定，Transformer使用了score归一化，即除以$\sqrt{dk}$；

5.对score施以softmax激活函数；

6.softmax点乘Value值v，得到加权的每个输入向量的评分v；

7.相加之后得到最终的输出结果z ：z = $\sum_{}^{}{}$ v。

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路，我们看个例子： The animal didn't cross the street because it was too tired 这里的it到底代表的是animal还是street呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把it和animal联系起来，接下来我们看下详细的处理过程。

1、首先，self-attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64低于embedding维度的。
![self attention](https://pic3.zhimg.com/80/v2-e473200fb3a2a00ce7467967d174ac76_1440w.jpg#pic_center)

那么Query、Key、Value这三个向量又是什么呢？这三个向量对于attention来说很重要，当你理解了下文后，你将会明白这三个向量扮演者什么的角色。

2、计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点乘，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2

![self attention q1·k2](https://pic3.zhimg.com/80/v2-8d98509cd1e0c7f72a0555c00cb8da06_1440w.jpg#pic_center)


3、接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大

![self attention softmax](https://pic3.zhimg.com/80/v2-41384c3fad61e1943466f5b6d2476c0a_1440w.jpg#pic_center)

4、下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值

![self attention value](https://pic2.zhimg.com/80/v2-87c41175e574e446b19334520f76b9bd_1440w.jpg#pic_center)


在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵Q与K相乘，乘以一个常数，做softmax操作，最后乘上V矩阵

![self attention values](https://pic4.zhimg.com/80/v2-3650acdf0c697e29aed2e0f01883cf2f_1440w.jpg#pic_center)

这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention：
![dot-product attention](https://pic1.zhimg.com/80/v2-e1c22655993d4a226c0f6353595398c4_1440w.jpg#pic_center)

以上就是self-attention的计算过程，下边是两个句子中it与上下文单词的关系热点图，很容易看出来第一个图片中的it与animal关系很强，第二个图it与street关系很强。这个结果说明注意力机制是可以很好地学习到上下文的语言信息。


<p align="center"><img width="400" src="https://pic2.zhimg.com/80/v2-a1f123cabab768b6c5cd9083aa22d68d_1440w.jpg#pic_center" title="self attention"></p>

### 5.注意力机制的优缺点

attention的优点

**1.参数少**：相比于 CNN、RNN ，其复杂度更小，参数也更少。所以对算力的要求也就更小。

**2.速度快**：Attention 解决了 RNN及其变体模型 不能并行计算的问题。Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。

**3.效果好**：在Attention 机制引入之前，有一个问题大家一直很苦恼：长距离的信息会被弱化，就好像记忆能力弱的人，记不住过去的事情是一样的。

attention的缺点：
需要的数据量大。因为注意力机制是抓重点信息，忽略不重要的信息，所以数据少的时候，注意力机制效果不如bilstm，现在我们企业都用注意力机制，因为企业数据都是十万百万级的数据量，用注意力机制就很好。还有传统的lstm，bilstm序列短的时候效果也比注意力机制好。所以注意力机制诞生的原因就是面向现在大数据的时代，企业里面动不动就是百万数据，超长序列，用传统的递归神经网络计算费时还不能并行计算。







