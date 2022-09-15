<!--
 * @Author: kavinbj
 * @Date: 2022-09-14 00:57:33
 * @LastEditTime: 2022-09-14 21:21:50
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->
# DeiT 学习笔记

### 《Training data-efficient image transformers & distillation through attention》
论文：https://arxiv.org/abs/2012.12877v2

代码：https://github.com/facebookresearch/deit


### ViT的限制
1. 需要海量的数据（google 私有数据 JFT-300M 3亿）
2. 需要庞大的计算资源


### DeiT主要贡献
1. DeiT无需海量预训练数据，只依靠ImageNet数据，便可以达到SOTA的结果，同时依赖的训练资源更少（4 GPUs in three days）
2. 仅使用 Transformer，不引入 Conv 的情况下也能达到 SOTA 效果
3. 提出了基于 token 蒸馏的策略，针对 Transformer 蒸馏方法超越传统蒸馏方法。


### 实现方法
1. 采用合适的训练策略包括optimizer,  regularization，（Better Hyperparameter更好的超参数设置），保证模型更好收敛。
2. data augmentation， 可以使用小数据训练
3. 采用distillation知识蒸馏的方式，结合teacher model来引导基于Transformer的DeiT更好地学习。进一步提高性能。
还有一些其他的方式，如：warmup、label smoothing、droppath等。

Performance comparison of DeiT and ViT
DeiT-B Distilled(1k ep)  83.4(84.2)
ViT-B 384                77.9

# Knowledge distillation
简单来说就是用teacher模型去训练student模型，通常teacher模型更大而且已经训练好了，student模型是我们当前需要训练的模型。在这个过程中，teacher模型是不训练的。
![Knowledge distillation](https://pic2.zhimg.com/80/v2-9f843f042ce6b4df0a56ee5a6f46826d_1440w.jpg)

当teacher模型和student模型拿到相同的图片时，都进行各自的前向，这时teacher模型就拿到了具有分类信息的feature，在进行softmax之前先除以一个参数τ，叫做temperature(蒸馏温度)，然后softmax得到soft labels(区别于one-hot形式的hard-label)。

student模型也是除以同一个τ，然后softmax得到一个soft-prediction，我们希望student模型的soft-prediction和teacher模型的soft labels尽量接近，使用KLDivLoss(2.1讲解)进行两者之间的差距度量，计算一个对应的损失teacher loss。

在训练的时候，我们是可以拿的到训练图片的真实的ground truth(hard label)的，可以看到上面图中student模型下面一路，就是预测结果和真是标签之间计算交叉熵crossentropy。

然后两路计算的损失：KLDivLoss和CELoss，按照一个加权关系计算得到一个总损失total loss，反向修改参数的时候这个teacher模型是不做训练的，只依据total loss训练student模型。

还可以使用硬蒸馏，对比上面的结构图，哪种更好没有定论。
![Knowledge distillation](https://pic1.zhimg.com/80/v2-a8672f44f5563ce6d5cad3b03438238c_1440w.jpg)

### KLDivloss
KL散度，又叫相对熵，用于衡量两个分布(连续分布和离散分布)之间的距离，在knowledge distillation中，两个分布为teacher模型和student模型的softmax输出。

![KLDivloss](https://pic1.zhimg.com/80/v2-c1ac2d6bd0de95764bffbd6a73443554_1440w.jpg)

想象一下，当两个分布很相近时候，对应class的预测值就会很接近，取log之后的差值就会很小，KL散度就很小。

当两个分布完全一致时候，KL散度就等于0。KLDivloss定义和使用方式为：
![KLDivloss2](https://pic1.zhimg.com/80/v2-e3520e9ffcdd50d98213575e27e66784_1440w.jpg)

### temperature
蒸馏温度τ 的作用，回想之前VIT中在self-attention里面计算qk间的加权因子的时候，计算完了要scale(除以k的维度)，然后再做softmax，然后用它们对v加权相加得到对应的表示向量。

如果是[1.0，20.0，400.0]直接做softamx，那结果是[0.0，0.0，1.0]，可见结果完全借鉴第三个引子。而先进行处理(比如除以1000)后变为[0.001，0.02，0.4]时，在做softamx结果为[0.28，0.29，0.42]结果总综合考虑了三部分，这显然是更合理的结果。实际中，看我是更希望结果偏向于更大的值，还是偏向于综合考虑来决定是否使用softmax前输入的预处理。


### distillation in transformer
![distillation in transformer](https://pic2.zhimg.com/80/v2-7d8c66dbb0562dfe6bf38b29850a407d_1440w.jpg)

在VIT中时使用class tokens去做分类的，相当于是一个额外的patch，这个patch去学习和别的patch之间的关系，然后连classifier，计算CELoss。在DeiT中为了做蒸馏，又额外加一个distill token，这个distill token也是去学和其他tokens之间的关系，然后连接teacher model计算KLDivLoss，那CELoss和KLDivLoss共同加权组合成一个新的loss取指导student model训练(知识蒸馏中teacher model不训练)。

在预测阶段，class token和distill token分别产生一个结果，然后将其加权(分别0.5)，再加在一起，得到最终的结果做预测。

### soft distillation vs hard distillation



## better hyperparameter
DeiT中第二个优化点在于better hyperparameter，也就是更好的参数配置，看看其都包含哪些部分。
![better hyperparameter](https://pic3.zhimg.com/80/v2-1d4db844edfaa1c354829d485fdca412_1440w.jpg)

参数初始化方式：truncated normal distribution(截断标准分布)。

learning-rate：CNN中的结论：当batch size越大的时候，learning rate设置的越大。

learning rate decay：cosine，在warm-up阶段lr先线性升上去，然后通过余弦方式lr降下来。

![better hyperparameter](https://pic3.zhimg.com/80/v2-e686aa5ad4cbbb8d2879f8bb283b6e1e_1440w.jpg)


## data augmentation
![data augmentation](https://pic4.zhimg.com/80/v2-136535b2629b711fb99db94f850c966b_1440w.jpg)

mixup之后的图片的label不再是单一的label，而是soft-label，比如[cat,dog]=[0.5,0.5]

cutmix之后的图片label是按所占据的比例给的，比如[cat,dog]=[0.3,0.7]

![data augmentation2](https://pic1.zhimg.com/80/v2-7fc994fb5b9941870373ba148607543c_1440w.jpg)

randomaug其实是由autoaug来的，autoaug是选取了25中增强策略，每种策略中有两个操作，这两种操作都要被执行。每次为一张图随机从25中策略中选取一种，将这两种操作对该图执行。至于这25中策略是怎么组成的，每种里面的操作的概率是如何确立的，这些是由搜索算法的实现的，总之认为这么搭配有效就行了。对于randomaug，相当于对于autoaug的简化，它是13种增强策略，然后从中一次选取6种策略依次对图片进行操作，完成增强操作。

model EMA(Exponential Moving Average)指数滑动平均，使得模型权重更新与一段时间内的历史取值有关。  是当前的模型权重，  是上一轮模型权重，  为模型当前权重的值，举一个例子：


![data augmentation2](https://pic2.zhimg.com/80/v2-1ad01556bf2fd23ceb9e84bff0847649_1440w.jpg)

![data augmentation3](https://pic4.zhimg.com/80/v2-85a55269648e157a9e8f90d771c63907_1440w.jpg)

三种更新参数方式的更新参数结果曲线：

![data augmentation4](https://pic4.zhimg.com/80/v2-888f16fcef8d023a0d00d0b71e053977_1440w.jpg)

![data augmentation5](https://pic4.zhimg.com/80/v2-068e9b0ac222e7ea264f798f3c68b5f7_1440w.jpg)

实际使用的时候，设置上面例子中的  值例如为0.99996，保证模型的参数值不会乱动。


### label smoothing
原本hard-label变成soft-label，设置参数，给其余非标签平均一些label概率。
![label smoothing](https://pic4.zhimg.com/80/v2-856b402786b5fe5b38ade59e09d38c4f_1440w.jpg)


## DeiT实现

![DeiT](https://pic2.zhimg.com/80/v2-bb013786fa17b744f73c7af31d1fb61d_1440w.jpg)


有了之前VIT的实现，可以发现其实DeiT是实现VIT和一个distillation token，以及一些调整性的操作。

1、为了实现蒸馏操作，加入一个新的distill token，输入是embed_dim，输出是num_classes。

2、返回结果在两个阶段不同，训练时候分别返回，以方便计算KLDivLoss、CELoss；推理时，网络参数已经训练好，这时候只需要返回两部分加权结果作为最终的判别结果就行了。

3、新增加的distill token和之前的class token大体相同，区别在于参数初始化方式不一样。

4、由于加入了distill token，因此给每个token加入position embedding时需要的个数是n+2。

patch embedding的前向过程：

1、在定义class token、distill token的时候我本并不知道输入的batch是多少，所以这里需要把第一个维度变为batch，这样的话就为每一个图片都增加了一个class token和distill token。

2、concat区别于add操作，是将三类token并在一起。(class一个、distill一个、imageN个)






